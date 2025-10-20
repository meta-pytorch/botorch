#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from abc import ABC
from copy import deepcopy
from functools import partial
from itertools import product
from random import randint

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import UserInputWarning
from botorch.models.transforms.input import (
    AffineInputTransform,
    AppendFeatures,
    BatchBroadcastedInputTransform,
    ChainedInputTransform,
    FilterFeatures,
    InputPerturbation,
    InputStandardize,
    InputTransform,
    InteractionFeatures,
    Log10,
    Normalize,
    NumericToCategoricalEncoding,
    OneHotToNumeric,
    ReversibleInputTransform,
    Round,
    Warp,
)
from botorch.models.transforms.utils import expand_and_copy_tensor
from botorch.models.utils import fantasize
from botorch.utils.testing import BotorchTestCase
from gpytorch import Module as GPyTorchModule
from gpytorch.priors import LogNormalPrior
from torch import Tensor
from torch.distributions import Kumaraswamy
from torch.nn import Module
from torch.nn.functional import one_hot


def get_test_warp(d, indices, bounds=None, **kwargs):
    if bounds is None:
        bounds = torch.zeros(2, d)
        bounds[1] = 1
    warp_tf = Warp(d=d, indices=indices, bounds=bounds, **kwargs)
    c0 = torch.tensor([1.0, 2.0])[: len(indices)]
    c1 = torch.tensor([2.0, 3.0])[: len(indices)]
    batch_shape = kwargs.get("batch_shape", torch.Size([]))
    c0 = c0.expand(batch_shape + c0.shape)
    c1 = c1.expand(batch_shape + c1.shape)
    warp_tf._set_concentration(0, c0)
    warp_tf._set_concentration(1, c1)
    return warp_tf


class NotSoAbstractInputTransform(InputTransform, Module):
    def __init__(  # noqa: D107
        self,
        transform_on_train,
        transform_on_eval,
        transform_on_fantasize=True,
    ):
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X):
        return X + 1


class TestInputTransforms(BotorchTestCase):
    def test_abstract_base_input_transform(self) -> None:
        with self.assertRaises(TypeError):
            InputTransform()
        X = torch.zeros([1])
        X_tf = torch.ones([1])
        # test transform_on_train and transform_on_eval
        ipt = NotSoAbstractInputTransform(
            transform_on_train=True, transform_on_eval=True
        )
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt = NotSoAbstractInputTransform(
            transform_on_train=True, transform_on_eval=False
        )
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X))
        ipt = NotSoAbstractInputTransform(
            transform_on_train=False, transform_on_eval=True
        )
        self.assertTrue(torch.equal(ipt(X), X))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X_tf))
        ipt = NotSoAbstractInputTransform(
            transform_on_train=False, transform_on_eval=False
        )
        self.assertTrue(torch.equal(ipt(X), X))
        ipt.eval()
        self.assertTrue(torch.equal(ipt(X), X))

        # test equals
        ipt2 = NotSoAbstractInputTransform(
            transform_on_train=False, transform_on_eval=False
        )
        self.assertTrue(ipt.equals(ipt2))
        ipt3 = NotSoAbstractInputTransform(
            transform_on_train=True, transform_on_eval=False
        )
        self.assertFalse(ipt.equals(ipt3))

        with self.assertRaises(NotImplementedError):
            ipt.untransform(None)

        # test preprocess_transform
        ipt4 = NotSoAbstractInputTransform(
            transform_on_train=True,
            transform_on_eval=False,
        )
        self.assertTrue(torch.equal(ipt4.preprocess_transform(X), X_tf))
        ipt4.transform_on_train = False
        self.assertTrue(torch.equal(ipt4.preprocess_transform(X), X))

        # test transform_on_fantasize
        ipt5 = NotSoAbstractInputTransform(
            transform_on_train=True, transform_on_eval=True, transform_on_fantasize=True
        )
        ipt5.eval()
        self.assertTrue(torch.equal(ipt5(X), X_tf))
        with fantasize():
            self.assertTrue(torch.equal(ipt5(X), X_tf))
        ipt5.transform_on_fantasize = False
        self.assertTrue(torch.equal(ipt5(X), X_tf))
        with fantasize():
            self.assertTrue(torch.equal(ipt5(X), X))

        # testing one line of AffineInputTransform
        # that doesn't have coverage otherwise
        d = 3
        coefficient, offset = torch.ones(d), torch.zeros(d)
        affine = AffineInputTransform(d, coefficient, offset)
        X = torch.randn(2, d)
        with self.assertRaises(NotImplementedError):
            affine._update_coefficients(X)

    def test_normalize(self) -> None:
        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 1000))
        for dtype in (torch.float, torch.double):
            # basic init, learned bounds
            nlz = Normalize(d=2)
            self.assertFalse(nlz.is_one_to_many)
            self.assertTrue(nlz.learn_bounds)
            self.assertTrue(nlz.training)
            self.assertEqual(nlz._d, 2)
            self.assertEqual(nlz.mins.shape, torch.Size([1, 2]))
            self.assertEqual(nlz.ranges.shape, torch.Size([1, 2]))
            nlz = Normalize(d=2, batch_shape=torch.Size([3]))
            self.assertTrue(nlz.learn_bounds)
            self.assertTrue(nlz.training)
            self.assertEqual(nlz._d, 2)
            self.assertEqual(nlz.mins.shape, torch.Size([3, 1, 2]))
            self.assertEqual(nlz.ranges.shape, torch.Size([3, 1, 2]))
            self.assertTrue(nlz.equals(Normalize(**nlz.get_init_args())))

            # learn_bounds=False with no bounds.
            with self.assertWarnsRegex(UserInputWarning, "learn_bounds"):
                Normalize(d=2, learn_bounds=False)

            # learn_bounds=True with bounds provided.
            bounds = torch.zeros(2, 2, device=self.device, dtype=dtype)
            nlz = Normalize(d=2, bounds=bounds, learn_bounds=True)
            self.assertTrue(nlz.learn_bounds)
            self.assertTrue(torch.equal(nlz.mins, bounds[..., 0:1, :]))
            self.assertTrue(
                torch.equal(nlz.ranges, bounds[..., 1:2, :] - bounds[..., 0:1, :])
            )

            # basic init, fixed bounds
            nlz = Normalize(d=2, bounds=bounds)
            self.assertFalse(nlz.learn_bounds)
            self.assertTrue(nlz.training)
            self.assertEqual(nlz._d, 2)
            self.assertTrue(torch.equal(nlz.mins, bounds[..., 0:1, :]))
            self.assertTrue(
                torch.equal(nlz.ranges, bounds[..., 1:2, :] - bounds[..., 0:1, :])
            )
            # with grad
            bounds.requires_grad = True
            bounds = bounds * 2
            self.assertIsNotNone(bounds.grad_fn)
            nlz = Normalize(d=2, bounds=bounds)
            # Set learn_coefficients=True for testing.
            nlz.learn_coefficients = True
            # We have grad in train mode.
            self.assertIsNotNone(nlz.coefficient.grad_fn)
            self.assertIsNotNone(nlz.offset.grad_fn)
            # Grad is detached in eval mode.
            nlz.eval()
            self.assertIsNone(nlz.coefficient.grad_fn)
            self.assertIsNone(nlz.offset.grad_fn)
            self.assertTrue(nlz.equals(Normalize(**nlz.get_init_args())))

            # basic init, provided indices
            with self.assertRaises(ValueError):
                nlz = Normalize(d=2, indices=[0, 1, 2])
            with self.assertRaises(ValueError):
                nlz = Normalize(d=2, indices=[0, 2])
            with self.assertRaises(ValueError):
                nlz = Normalize(d=2, indices=[0, 0])
            with self.assertRaises(ValueError):
                nlz = Normalize(d=2, indices=[])
            nlz = Normalize(d=2, indices=[0])
            self.assertTrue(nlz.learn_bounds)
            self.assertTrue(nlz.training)
            self.assertEqual(nlz._d, 2)
            self.assertEqual(nlz.mins.shape, torch.Size([1, 1]))
            self.assertEqual(nlz.ranges.shape, torch.Size([1, 1]))
            self.assertEqual(len(nlz.indices), 1)
            nlz.to(device=self.device)
            self.assertTrue(
                (
                    nlz.indices
                    == torch.tensor([0], dtype=torch.long, device=self.device)
                ).all()
            )
            self.assertTrue(nlz.equals(Normalize(**nlz.get_init_args())))

            # test .to
            other_dtype = torch.float if dtype == torch.double else torch.double
            nlz.to(other_dtype)
            self.assertTrue(nlz.mins.dtype == other_dtype)
            # test incompatible dimensions of specified bounds
            bounds = torch.zeros(2, 3, device=self.device, dtype=dtype)
            with self.assertRaisesRegex(
                BotorchTensorDimensionError,
                "Dimensions of provided `bounds` are incompatible",
            ):
                Normalize(d=2, bounds=bounds)

            # test jitter
            nlz = Normalize(d=2, min_range=1e-4)
            self.assertEqual(nlz.min_range, 1e-4)
            X = torch.cat((torch.randn(4, 1), torch.zeros(4, 1)), dim=-1)
            X = X.to(self.device)
            self.assertEqual(torch.isfinite(nlz(X)).sum(), X.numel())
            with self.assertRaisesRegex(
                BotorchTensorDimensionError, r"must have at least 2 dimensions"
            ):
                nlz(torch.randn(X.shape[-1], dtype=dtype))

            # using unbatched X to train batched transform
            nlz = Normalize(d=2, min_range=1e-4, batch_shape=torch.Size([3]))
            X = torch.rand(4, 2)
            with self.assertRaisesRegex(
                ValueError, "must have at least 3 dimensions, 1 batch and 2 innate"
            ):
                nlz(X)

            # basic usage
            for batch_shape, center in product(
                (torch.Size(), torch.Size([3])), [0.5, 0.0]
            ):
                # learned bounds
                nlz = Normalize(d=2, batch_shape=batch_shape, center=center)
                X = torch.randn(*batch_shape, 4, 2, device=self.device, dtype=dtype)
                for _X in (torch.stack((X, X)), X):  # check batch_shape is obeyed
                    X_nlzd = nlz(_X)
                    self.assertEqual(nlz.mins.shape, batch_shape + (1, X.shape[-1]))
                    self.assertEqual(nlz.ranges.shape, batch_shape + (1, X.shape[-1]))

                self.assertAllClose(X_nlzd.min().item(), center - 0.5)
                self.assertAllClose(X_nlzd.max().item(), center + 0.5)

                nlz.eval()
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertAllClose(X, X_unnlzd, atol=1e-3, rtol=1e-3)
                expected_bounds = torch.cat(
                    [X.min(dim=-2, keepdim=True)[0], X.max(dim=-2, keepdim=True)[0]],
                    dim=-2,
                )
                coeff = expected_bounds[..., 1, :] - expected_bounds[..., 0, :]
                expected_bounds[..., 0, :] += (0.5 - center) * coeff
                expected_bounds[..., 1, :] = expected_bounds[..., 0, :] + coeff
                atol = 1e-6 if dtype is torch.float32 else 1e-12
                rtol = 1e-4 if dtype is torch.float32 else 1e-8
                self.assertAllClose(nlz.bounds, expected_bounds, atol=atol, rtol=rtol)
                # test errors on wrong shape
                nlz = Normalize(d=2, batch_shape=batch_shape)
                X = torch.randn(*batch_shape, 2, 1, device=self.device, dtype=dtype)
                expected_msg = "Wrong input dimension. Received 1, expected 2."
                with self.assertRaisesRegex(BotorchTensorDimensionError, expected_msg):
                    nlz(X)
                # Same error in eval mode
                nlz.eval()
                with self.assertRaisesRegex(BotorchTensorDimensionError, expected_msg):
                    nlz(X)

                # fixed bounds
                bounds = torch.tensor(
                    [[-2.0, -1], [1, 2.0]], device=self.device, dtype=dtype
                ).expand(*batch_shape, 2, 2)
                nlz = Normalize(d=2, bounds=bounds)
                X = torch.rand(*batch_shape, 4, 2, device=self.device, dtype=dtype)
                X_nlzd = nlz(X)
                self.assertTrue(torch.equal(nlz.bounds, bounds))
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertAllClose(X, X_unnlzd, atol=1e-4, rtol=1e-4)

                # test no normalization on eval
                nlz = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, transform_on_eval=False
                )
                X_nlzd = nlz(X)
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertAllClose(X, X_unnlzd, atol=1e-4, rtol=1e-4)
                nlz.eval()
                self.assertTrue(torch.equal(nlz(X), X))

                # test no normalization on train
                nlz = Normalize(
                    d=2,
                    bounds=bounds,
                    batch_shape=batch_shape,
                    transform_on_train=False,
                )
                X_nlzd = nlz(X)
                self.assertTrue(torch.equal(nlz(X), X))
                nlz.eval()
                X_nlzd = nlz(X)
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertAllClose(X, X_unnlzd, atol=1e-4, rtol=1e-4)

                # test reverse
                nlz = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, reverse=True
                )
                X2 = nlz(X_nlzd)
                self.assertAllClose(X2, X, atol=1e-4, rtol=1e-4)
                X_nlzd2 = nlz.untransform(X2)
                self.assertAllClose(X_nlzd, X_nlzd2, atol=1e-4, rtol=1e-4)

                # test non complete indices
                indices = [0, 2]
                nlz = Normalize(d=3, batch_shape=batch_shape, indices=indices)
                X = torch.randn(*batch_shape, 4, 3, device=self.device, dtype=dtype)
                X_nlzd = nlz(X)
                self.assertEqual(X_nlzd[..., indices].min().item(), 0.0)
                self.assertEqual(X_nlzd[..., indices].max().item(), 1.0)
                self.assertAllClose(X_nlzd[..., 1], X[..., 1])
                nlz.eval()
                X_unnlzd = nlz.untransform(X_nlzd)
                self.assertAllClose(X, X_unnlzd, atol=1e-4, rtol=1e-4)
                expected_bounds = torch.cat(
                    [X.min(dim=-2, keepdim=True)[0], X.max(dim=-2, keepdim=True)[0]],
                    dim=-2,
                )[..., indices]
                self.assertAllClose(nlz.bounds, expected_bounds, atol=1e-4, rtol=1e-4)

                # test errors on wrong shape
                nlz = Normalize(d=2, batch_shape=batch_shape)
                X = torch.randn(*batch_shape, 2, 1, device=self.device, dtype=dtype)
                with self.assertRaisesRegex(
                    BotorchTensorDimensionError,
                    "Wrong input dimension. Received 1, expected 2.",
                ):
                    nlz(X)

                # test equals
                nlz = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, reverse=True
                )
                nlz2 = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, reverse=False
                )
                self.assertFalse(nlz.equals(nlz2))
                nlz3 = Normalize(
                    d=2, bounds=bounds, batch_shape=batch_shape, reverse=True
                )
                self.assertTrue(nlz.equals(nlz3))
                new_bounds = bounds + 1
                nlz4 = Normalize(
                    d=2, bounds=new_bounds, batch_shape=batch_shape, reverse=True
                )
                self.assertFalse(nlz.equals(nlz4))
                nlz5 = Normalize(d=2, batch_shape=batch_shape)
                self.assertFalse(nlz.equals(nlz5))
                nlz6 = Normalize(d=2, batch_shape=batch_shape, indices=[0, 1])
                self.assertFalse(nlz5.equals(nlz6))
                nlz7 = Normalize(d=2, batch_shape=batch_shape, indices=[0])
                self.assertFalse(nlz5.equals(nlz7))
                nlz8 = Normalize(d=2, batch_shape=batch_shape, indices=[0, 1])
                self.assertTrue(nlz6.equals(nlz8))
                nlz9 = Normalize(d=3, batch_shape=batch_shape, indices=[0, 1])
                nlz10 = Normalize(d=3, batch_shape=batch_shape, indices=[0, 2])
                self.assertFalse(nlz9.equals(nlz10))

                # test with grad
                nlz = Normalize(d=1)
                X.requires_grad = True
                X = X * 2
                self.assertIsNotNone(X.grad_fn)
                nlz(X)
                self.assertIsNotNone(nlz.coefficient.grad_fn)
                self.assertIsNotNone(nlz.offset.grad_fn)
                nlz.eval()
                self.assertIsNone(nlz.coefficient.grad_fn)
                self.assertIsNone(nlz.offset.grad_fn)

            # test that zero range is not scaled.
            nlz = Normalize(d=2)
            X = torch.tensor([[1.0, 0.0], [1.0, 2.0]], device=self.device, dtype=dtype)
            nlzd_X = nlz(X)
            self.assertAllClose(
                nlz.coefficient,
                torch.tensor([[1.0, 2.0]], device=self.device, dtype=dtype),
            )
            expected_X = torch.tensor(
                [[1.0, 0.0], [1.0, 1.0]], device=self.device, dtype=dtype
            )
            self.assertAllClose(nlzd_X, expected_X)
            nlz.eval()
            X = torch.tensor([[1.5, 1.5]], device=self.device, dtype=dtype)
            nlzd_X = nlz(X)
            expected_X = torch.tensor([[1.5, 0.75]], device=self.device, dtype=dtype)
            self.assertAllClose(nlzd_X, expected_X)

            # Test broadcasting across batch dimensions in eval mode
            x = torch.tensor(
                [[0.0, 2.0], [3.0, 5.0]], device=self.device, dtype=dtype
            ).unsqueeze(-1)
            self.assertEqual(x.shape, torch.Size([2, 2, 1]))
            nlz = Normalize(d=1, batch_shape=torch.Size([2]))
            nlz(x)
            nlz.eval()
            x2 = torch.tensor([[1.0]], device=self.device, dtype=dtype)
            nlzd_x2 = nlz.transform(x2)
            self.assertEqual(nlzd_x2.shape, torch.Size([2, 1, 1]))
            self.assertAllClose(
                nlzd_x2.squeeze(),
                torch.tensor([0.5, -1.0], dtype=dtype, device=self.device),
            )

    def test_standardize(self) -> None:
        for dtype in (torch.float, torch.double):
            # basic init
            stdz = InputStandardize(d=2)
            self.assertTrue(stdz.training)
            self.assertEqual(stdz._d, 2)
            self.assertEqual(stdz.means.shape, torch.Size([1, 2]))
            self.assertEqual(stdz.stds.shape, torch.Size([1, 2]))
            stdz = InputStandardize(d=2, batch_shape=torch.Size([3]))
            self.assertTrue(stdz.training)
            self.assertEqual(stdz._d, 2)
            self.assertEqual(stdz.means.shape, torch.Size([3, 1, 2]))
            self.assertEqual(stdz.stds.shape, torch.Size([3, 1, 2]))

            # basic init, provided indices
            with self.assertRaises(ValueError):
                stdz = InputStandardize(d=2, indices=[0, 1, 2])
            with self.assertRaises(ValueError):
                stdz = InputStandardize(d=2, indices=[0, 2])
            with self.assertRaises(ValueError):
                stdz = InputStandardize(d=2, indices=[0, 0])
            with self.assertRaises(ValueError):
                stdz = InputStandardize(d=2, indices=[])
            stdz = InputStandardize(d=2, indices=[0])
            self.assertTrue(stdz.training)
            self.assertEqual(stdz._d, 2)
            self.assertEqual(stdz.means.shape, torch.Size([1, 1]))
            self.assertEqual(stdz.stds.shape, torch.Size([1, 1]))
            self.assertEqual(len(stdz.indices), 1)
            stdz.to(device=self.device)
            self.assertTrue(
                torch.equal(
                    stdz.indices,
                    torch.tensor([0], dtype=torch.long, device=self.device),
                )
            )
            stdz = InputStandardize(d=2, indices=[0], batch_shape=torch.Size([3]))
            stdz.to(device=self.device)
            self.assertTrue(stdz.training)
            self.assertEqual(stdz._d, 2)
            self.assertEqual(stdz.means.shape, torch.Size([3, 1, 1]))
            self.assertEqual(stdz.stds.shape, torch.Size([3, 1, 1]))
            self.assertEqual(len(stdz.indices), 1)
            self.assertTrue(
                torch.equal(
                    stdz.indices,
                    torch.tensor([0], device=self.device, dtype=torch.long),
                )
            )

            # test jitter
            stdz = InputStandardize(d=2, min_std=1e-4)
            self.assertEqual(stdz.min_std, 1e-4)
            X = torch.cat((torch.randn(4, 1), torch.zeros(4, 1)), dim=-1)
            X = X.to(self.device, dtype=dtype)
            self.assertEqual(torch.isfinite(stdz(X)).sum(), X.numel())
            with self.assertRaisesRegex(
                BotorchTensorDimensionError, r"must have at least \d+ dim"
            ):
                stdz(torch.randn(X.shape[-1], dtype=dtype))

            # basic usage
            for batch_shape in (torch.Size(), torch.Size([3])):
                stdz = InputStandardize(d=2, batch_shape=batch_shape)
                torch.manual_seed(42)
                X = torch.randn(*batch_shape, 4, 2, device=self.device, dtype=dtype)
                for _X in (torch.stack((X, X)), X):  # check batch_shape is obeyed
                    X_stdz = stdz(_X)
                    self.assertEqual(stdz.means.shape, batch_shape + (1, X.shape[-1]))
                    self.assertEqual(stdz.stds.shape, batch_shape + (1, X.shape[-1]))

                self.assertTrue(torch.all(X_stdz.mean(dim=-2).abs() < 1e-4))
                self.assertTrue(torch.all((X_stdz.std(dim=-2) - 1.0).abs() < 1e-4))

                stdz.eval()
                X_unstdz = stdz.untransform(X_stdz)
                self.assertAllClose(X, X_unstdz, atol=1e-4, rtol=1e-4)
                expected_means = X.mean(dim=-2, keepdim=True)
                expected_stds = X.std(dim=-2, keepdim=True)
                self.assertAllClose(stdz.means, expected_means)
                self.assertAllClose(stdz.stds, expected_stds)

                # test to
                other_dtype = torch.float if dtype == torch.double else torch.double
                stdz.to(other_dtype)
                self.assertTrue(stdz.means.dtype == other_dtype)

                # test errors on wrong shape
                stdz = InputStandardize(d=2, batch_shape=batch_shape)
                X = torch.randn(*batch_shape, 2, 1, device=self.device, dtype=dtype)
                with self.assertRaises(BotorchTensorDimensionError):
                    stdz(X)

                # test no normalization on eval
                stdz = InputStandardize(
                    d=2, batch_shape=batch_shape, transform_on_eval=False
                )
                X = torch.randn(*batch_shape, 4, 2, device=self.device, dtype=dtype)
                X_stdz = stdz(X)
                X_unstdz = stdz.untransform(X_stdz)
                self.assertAllClose(X, X_unstdz, atol=1e-4, rtol=1e-4)
                stdz.eval()
                self.assertTrue(torch.equal(stdz(X), X))

                # test no normalization on train
                stdz = InputStandardize(
                    d=2, batch_shape=batch_shape, transform_on_train=False
                )
                X_stdz = stdz(X)
                self.assertTrue(torch.equal(stdz(X), X))
                stdz.eval()
                X_unstdz = stdz.untransform(X_stdz)
                self.assertAllClose(X, X_unstdz, atol=1e-4, rtol=1e-4)

                # test indices
                indices = [0, 2]
                stdz = InputStandardize(d=3, batch_shape=batch_shape, indices=indices)
                X = torch.randn(*batch_shape, 4, 3, device=self.device, dtype=dtype)
                X_stdz = stdz(X)
                self.assertTrue(
                    torch.all(X_stdz[..., indices].mean(dim=-2).abs() < 1e-4)
                )
                self.assertTrue(
                    torch.all(X_stdz[..., indices].std(dim=-2) < 1.0 + 1e-4)
                )
                self.assertTrue(
                    torch.all((X_stdz[..., indices].std(dim=-2) - 1.0).abs() < 1e-4)
                )
                self.assertAllClose(X_stdz[..., 1], X[..., 1])
                stdz.eval()
                X_unstdz = stdz.untransform(X_stdz)
                self.assertAllClose(X, X_unstdz, atol=1e-4, rtol=1e-4)

                # test equals
                stdz = InputStandardize(d=2, batch_shape=batch_shape, reverse=True)
                stdz2 = InputStandardize(d=2, batch_shape=batch_shape, reverse=False)
                self.assertFalse(stdz.equals(stdz2))
                stdz3 = InputStandardize(d=2, batch_shape=batch_shape, reverse=True)
                self.assertTrue(stdz.equals(stdz3))
                stdz4 = InputStandardize(d=2, batch_shape=batch_shape, indices=[0, 1])
                self.assertFalse(stdz4.equals(stdz))
                stdz5 = InputStandardize(d=2, batch_shape=batch_shape, indices=[0])
                self.assertFalse(stdz5.equals(stdz))
                stdz6 = InputStandardize(d=2, batch_shape=batch_shape, indices=[0, 1])
                self.assertTrue(stdz6.equals(stdz4))
                stdz7 = InputStandardize(d=3, batch_shape=batch_shape, indices=[0, 1])
                stdz8 = InputStandardize(d=3, batch_shape=batch_shape, indices=[0, 2])
                self.assertFalse(stdz7.equals(stdz8))

    def test_chained_input_transform(self) -> None:
        ds = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)
        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 1000))

        for d, batch_shape, dtype in itertools.product(ds, batch_shapes, dtypes):
            bounds = torch.tensor(
                [[-2.0] * d, [2.0] * d], device=self.device, dtype=dtype
            )
            tf1 = Normalize(d=d, bounds=bounds, batch_shape=batch_shape)
            tf2 = Normalize(d=d, batch_shape=batch_shape)
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            # make copies for validation below
            tf1_, tf2_ = deepcopy(tf1), deepcopy(tf2)
            self.assertTrue(tf.training)
            self.assertEqual(sorted(tf.keys()), ["stz_fixed", "stz_learned"])
            self.assertEqual(tf["stz_fixed"], tf1)
            self.assertEqual(tf["stz_learned"], tf2)
            self.assertFalse(tf.is_one_to_many)

            X = torch.rand(*batch_shape, 4, d, device=self.device, dtype=dtype)
            X_tf = tf(X)
            X_tf_ = tf2_(tf1_(X))
            self.assertTrue(tf1.training)
            self.assertTrue(tf2.training)
            self.assertTrue(torch.equal(X_tf, X_tf_))
            X_utf = tf.untransform(X_tf)
            self.assertAllClose(X_utf, X, atol=1e-4, rtol=1e-4)

            # test not transformed on eval
            tf1.transform_on_eval = False
            tf2.transform_on_eval = False
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            tf.eval()
            self.assertTrue(torch.equal(tf(X), X))
            tf1.transform_on_eval = True
            tf2.transform_on_eval = True
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            tf.eval()
            self.assertTrue(torch.equal(tf(X), X_tf))

            # test not transformed on train
            tf1.transform_on_train = False
            tf2.transform_on_train = False
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            tf.train()
            self.assertTrue(torch.equal(tf(X), X))

            # test __eq__
            other_tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            self.assertTrue(tf.equals(other_tf))
            # change order
            other_tf = ChainedInputTransform(stz_learned=tf2, stz_fixed=tf1)
            self.assertFalse(tf.equals(other_tf))
            # Identical transforms but different objects.
            other_tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=deepcopy(tf2))
            self.assertTrue(tf.equals(other_tf))

            # test preprocess_transform
            tf2.transform_on_train = False
            tf1.transform_on_train = True
            tf = ChainedInputTransform(stz_fixed=tf1, stz_learned=tf2)
            self.assertTrue(torch.equal(tf.preprocess_transform(X), tf1.transform(X)))

        # test one-to-many
        tf2 = InputPerturbation(perturbation_set=bounds)
        tf = ChainedInputTransform(stz=tf1, pert=tf2)
        self.assertTrue(tf.is_one_to_many)

    def test_batch_broadcasted_input_transform(self) -> None:
        ds = (1, 2)
        batch_args = [
            (torch.Size([2]), {}),
            (torch.Size([3, 2]), {}),
            (torch.Size([2, 3]), {"broadcast_index": 0}),
            (torch.Size([5, 2, 3]), {"broadcast_index": 1}),
        ]
        dtypes = (torch.float, torch.double)
        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 1000))

        for d, (batch_shape, kwargs), dtype in itertools.product(
            ds, batch_args, dtypes
        ):
            bounds = torch.tensor(
                [[-2.0] * d, [2.0] * d], device=self.device, dtype=dtype
            )
            # when the batch_shape is (2, 3), the transform list is broadcasted across
            # the first dimension, whereas each individual transform gets broadcasted
            # over the remaining batch dimensions.
            if "broadcast_index" not in kwargs:
                broadcast_index = -3
                tf_batch_shape = batch_shape[:-1]
            else:
                broadcast_index = kwargs["broadcast_index"]
                # if the broadcast index is negative, we need to adjust the index
                # when indexing into the batch shape tuple
                i = broadcast_index + 2 if broadcast_index < 0 else broadcast_index
                tf_batch_shape = list(batch_shape[:i])
                tf_batch_shape.extend(list(batch_shape[i + 1 :]))
                tf_batch_shape = torch.Size(tf_batch_shape)

            tf1 = Normalize(d=d, bounds=bounds, batch_shape=tf_batch_shape)
            tf2 = InputStandardize(d=d, batch_shape=tf_batch_shape)
            transforms = [tf1, tf2]
            tf = BatchBroadcastedInputTransform(transforms=transforms, **kwargs)
            # make copies for validation below
            transforms_ = [deepcopy(tf_i) for tf_i in transforms]
            self.assertTrue(tf.training)
            # self.assertEqual(sorted(tf.keys()), ["stz_fixed", "stz_learned"])
            self.assertEqual(tf.transforms[0], tf1)
            self.assertEqual(tf.transforms[1], tf2)
            self.assertFalse(tf.is_one_to_many)

            X = torch.rand(*batch_shape, 4, d, device=self.device, dtype=dtype)
            X_tf = tf(X)
            Xs = X.unbind(dim=broadcast_index)

            X_tf_ = torch.stack(
                [tf_i_(Xi) for tf_i_, Xi in zip(transforms_, Xs)], dim=broadcast_index
            )
            self.assertTrue(tf1.training)
            self.assertTrue(tf2.training)
            self.assertTrue(torch.equal(X_tf, X_tf_))
            X_utf = tf.untransform(X_tf)
            self.assertAllClose(X_utf, X, atol=1e-4, rtol=1e-4)

            # test not transformed on eval
            for tf_i in transforms:
                tf_i.transform_on_eval = False

            tf = BatchBroadcastedInputTransform(transforms=transforms, **kwargs)
            tf.eval()
            self.assertTrue(torch.equal(tf(X), X))

            # test transformed on eval
            for tf_i in transforms:
                tf_i.transform_on_eval = True

            tf = BatchBroadcastedInputTransform(transforms=transforms, **kwargs)
            tf.eval()
            self.assertTrue(torch.equal(tf(X), X_tf))

            # test not transformed on train
            for tf_i in transforms:
                tf_i.transform_on_train = False

            tf = BatchBroadcastedInputTransform(transforms=transforms, **kwargs)
            tf.train()
            self.assertTrue(torch.equal(tf(X), X))

            # test __eq__
            other_tf = BatchBroadcastedInputTransform(transforms=transforms, **kwargs)
            self.assertTrue(tf.equals(other_tf))
            # change order
            other_tf = BatchBroadcastedInputTransform(
                transforms=list(reversed(transforms))
            )
            self.assertFalse(tf.equals(other_tf))
            # Identical transforms but different objects.
            other_tf = BatchBroadcastedInputTransform(
                transforms=deepcopy(transforms), **kwargs
            )
            self.assertTrue(tf.equals(other_tf))

            # test preprocess_transform
            transforms[-1].transform_on_train = False
            transforms[0].transform_on_train = True
            tf = BatchBroadcastedInputTransform(transforms=transforms, **kwargs)
            self.assertTrue(
                torch.equal(
                    tf.preprocess_transform(X).unbind(dim=broadcast_index)[0],
                    transforms[0].transform(Xs[0]),
                )
            )

        # test one-to-many
        tf2 = InputPerturbation(perturbation_set=2 * bounds)
        with self.assertRaisesRegex(ValueError, r".*one_to_many.*"):
            tf = BatchBroadcastedInputTransform(transforms=[tf1, tf2], **kwargs)

        # these could technically be batched internally, but we're testing the generic
        # batch broadcasted transform list here. Could change test to use AppendFeatures
        tf1 = InputPerturbation(perturbation_set=bounds)
        tf2 = InputPerturbation(perturbation_set=2 * bounds)
        tf = BatchBroadcastedInputTransform(transforms=[tf1, tf2], **kwargs)
        self.assertTrue(tf.is_one_to_many)

        with self.assertRaisesRegex(
            ValueError, r"The broadcast index cannot be -2 and -1"
        ):
            tf = BatchBroadcastedInputTransform(
                transforms=[tf1, tf2], broadcast_index=-2
            )

    def test_round_transform_init(self) -> None:
        # basic init
        int_idcs = [0, 4]
        categorical_feats = {2: 2, 5: 3}
        round_tf = Round(
            integer_indices=int_idcs, categorical_features=categorical_feats
        )
        self.assertEqual(round_tf.integer_indices.tolist(), int_idcs)
        self.assertEqual(round_tf.categorical_features, categorical_feats)
        self.assertTrue(round_tf.training)
        self.assertFalse(round_tf.approximate)
        self.assertEqual(round_tf.tau, 1e-3)
        self.assertTrue(round_tf.equals(Round(**round_tf.get_init_args())))

        for dtype in (torch.float, torch.double):
            # With tensor indices.
            round_tf = Round(
                integer_indices=torch.tensor(int_idcs, dtype=dtype, device=self.device),
                categorical_features=categorical_feats,
            )
            self.assertEqual(round_tf.integer_indices.tolist(), int_idcs)
            self.assertTrue(round_tf.equals(Round(**round_tf.get_init_args())))

    def test_round_transform(self) -> None:
        int_idcs = [0, 4]
        categorical_feats = {2: 2, 5: 3}
        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 1000))
        for dtype, batch_shape, approx, categorical_features in itertools.product(
            (torch.float, torch.double),
            (torch.Size(), torch.Size([3])),
            (False, True),
            (None, categorical_feats),
        ):
            with self.subTest(
                dtype=dtype,
                batch_shape=batch_shape,
                approx=approx,
                categorical_features=categorical_features,
            ):
                X = torch.rand(*batch_shape, 4, 8, device=self.device, dtype=dtype)
                X[..., int_idcs] *= 5
                if categorical_features is not None and approx:
                    with self.assertRaises(NotImplementedError):
                        Round(
                            integer_indices=int_idcs,
                            categorical_features=categorical_features,
                            approximate=approx,
                        )
                    continue
                tau = 1e-1
                round_tf = Round(
                    integer_indices=int_idcs,
                    categorical_features=categorical_features,
                    approximate=approx,
                    tau=tau,
                )
                X_rounded = round_tf(X)
                exact_rounded_X_ints = X[..., int_idcs].round()
                # check non-integers parameters are unchanged
                self.assertTrue(torch.equal(X_rounded[..., 1], X[..., 1]))
                if approx:
                    # check that approximate rounding is closer to rounded values than
                    # the original inputs
                    dist_approx_to_rounded = (
                        X_rounded[..., int_idcs] - exact_rounded_X_ints
                    ).abs()
                    dist_orig_to_rounded = (
                        X[..., int_idcs] - exact_rounded_X_ints
                    ).abs()
                    tol = torch.tanh(torch.tensor(0.5 / tau, dtype=dtype))
                    self.assertGreater(
                        (dist_orig_to_rounded - dist_approx_to_rounded).min(), -tol
                    )

                    self.assertFalse(
                        torch.equal(X_rounded[..., int_idcs], exact_rounded_X_ints)
                    )
                else:
                    # check that exact rounding behaves as expected for integers
                    self.assertTrue(
                        torch.equal(X_rounded[..., int_idcs], exact_rounded_X_ints)
                    )
                    if categorical_features is not None:
                        # test that discretization works as expected for categoricals
                        for start, card in categorical_features.items():
                            end = start + card
                            expected_categorical = one_hot(
                                X[..., start:end].argmax(dim=-1), num_classes=card
                            ).to(X)
                            self.assertTrue(
                                torch.equal(
                                    X_rounded[..., start:end], expected_categorical
                                )
                            )
                    # test that gradient information is passed via STE
                    X2 = X.clone().requires_grad_(True)
                    round_tf(X2).sum().backward()
                    self.assertTrue(torch.equal(X2.grad, torch.ones_like(X2)))
                with self.assertRaises(NotImplementedError):
                    round_tf.untransform(X_rounded)

                # test no transform on eval
                round_tf = Round(
                    integer_indices=int_idcs,
                    categorical_features=categorical_features,
                    approximate=approx,
                    transform_on_eval=False,
                )
                X_rounded = round_tf(X)
                self.assertFalse(torch.equal(X, X_rounded))
                round_tf.eval()
                X_rounded = round_tf(X)
                self.assertTrue(torch.equal(X, X_rounded))

                # test no transform on train
                round_tf = Round(
                    integer_indices=int_idcs,
                    categorical_features=categorical_features,
                    approximate=approx,
                    transform_on_train=False,
                )
                X_rounded = round_tf(X)
                self.assertTrue(torch.equal(X, X_rounded))
                round_tf.eval()
                X_rounded = round_tf(X)
                self.assertFalse(torch.equal(X, X_rounded))

                # test equals
                round_tf2 = Round(
                    integer_indices=int_idcs,
                    categorical_features=categorical_features,
                    approximate=approx,
                    transform_on_train=False,
                )
                self.assertTrue(round_tf.equals(round_tf2))
                # test different transform_on_train
                round_tf2 = Round(
                    integer_indices=int_idcs,
                    categorical_features=categorical_features,
                    approximate=approx,
                )
                self.assertFalse(round_tf.equals(round_tf2))
                # test different approx
                round_tf = Round(
                    integer_indices=int_idcs,
                )
                round_tf2 = Round(
                    integer_indices=int_idcs,
                    approximate=not approx,
                    transform_on_train=False,
                )
                self.assertFalse(round_tf.equals(round_tf2))
                # test different indices
                round_tf = Round(
                    integer_indices=int_idcs,
                    categorical_features=categorical_features,
                    transform_on_train=False,
                )
                round_tf2 = Round(
                    integer_indices=[0, 1],
                    categorical_features=categorical_features,
                    approximate=approx,
                    transform_on_train=False,
                )
                self.assertFalse(round_tf.equals(round_tf2))

                # test preprocess_transform
                round_tf.transform_on_train = False
                self.assertTrue(torch.equal(round_tf.preprocess_transform(X), X))
                round_tf.transform_on_train = True
                X_rounded = round_tf(X)
                self.assertTrue(
                    torch.equal(round_tf.preprocess_transform(X), X_rounded)
                )

    def test_log10_transform(self) -> None:
        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 1000))
        for dtype in (torch.float, torch.double):
            # basic init
            indices = [0, 2]
            log_tf = Log10(indices=indices)
            self.assertEqual(log_tf.indices.tolist(), indices)
            self.assertTrue(log_tf.training)

            # basic usage
            for batch_shape in (torch.Size(), torch.Size([3])):
                X = 1 + 5 * torch.rand(
                    *batch_shape, 4, 3, device=self.device, dtype=dtype
                )
                log_tf = Log10(indices=indices)
                X_tf = log_tf(X)
                expected_X_tf = X.clone()
                expected_X_tf[..., indices] = expected_X_tf[..., indices].log10()
                # check non-integers parameters are unchanged
                self.assertTrue(torch.equal(expected_X_tf, X_tf))
                untransformed_X = log_tf.untransform(X_tf)
                self.assertAllClose(untransformed_X, X)

                # test no transform on eval
                log_tf = Log10(indices=indices, transform_on_eval=False)
                X_tf = log_tf(X)
                self.assertFalse(torch.equal(X, X_tf))
                log_tf.eval()
                X_tf = log_tf(X)
                self.assertTrue(torch.equal(X, X_tf))

                # test no transform on train
                log_tf = Log10(indices=indices, transform_on_train=False)
                X_tf = log_tf(X)
                self.assertTrue(torch.equal(X, X_tf))
                log_tf.eval()
                X_tf = log_tf(X)
                self.assertFalse(torch.equal(X, X_tf))

                # test equals
                log_tf2 = Log10(indices=indices, transform_on_train=False)
                self.assertTrue(log_tf.equals(log_tf2))
                # test different transform_on_train
                log_tf2 = Log10(indices=indices)
                self.assertFalse(log_tf.equals(log_tf2))
                # test different indices
                log_tf2 = Log10(indices=[0, 1], transform_on_train=False)
                self.assertFalse(log_tf.equals(log_tf2))

                # test preprocess_transform
                log_tf.transform_on_train = False
                self.assertTrue(torch.equal(log_tf.preprocess_transform(X), X))
                log_tf.transform_on_train = True
                self.assertTrue(torch.equal(log_tf.preprocess_transform(X), X_tf))

    def test_warp_transform(self) -> None:
        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 1000))
        for dtype, batch_shape, warp_batch_shape in itertools.product(
            (torch.float, torch.double),
            (torch.Size(), torch.Size([3])),
            (torch.Size(), torch.Size([2])),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            eps = 1e-6 if dtype == torch.double else 1e-5
            if dtype == torch.float32:
                # defaults are 1e-5, 1e-8
                tols = {"rtol": 1e-4, "atol": 1e-6}
            else:
                tols = {}

            # basic init
            indices = [0, 2]
            warp_tf = get_test_warp(
                d=3, indices=indices, batch_shape=warp_batch_shape, eps=eps
            ).to(**tkwargs)
            self.assertTrue(warp_tf.training)

            k = Kumaraswamy(warp_tf.concentration1, warp_tf.concentration0)

            self.assertEqual(warp_tf.indices.tolist(), indices)

            # We don't want these data points to end up all the way near zero, since
            # this would cause numerical issues and thus result in a flaky test.
            X = 0.025 + 0.95 * torch.rand(*batch_shape, 4, 3, **tkwargs)
            X = X.unsqueeze(-3) if len(warp_batch_shape) > 0 else X
            with torch.no_grad():
                warp_tf = get_test_warp(
                    d=3, indices=indices, batch_shape=warp_batch_shape, eps=eps
                ).to(**tkwargs)
                X_tf = warp_tf(X)
                expected_X_tf = expand_and_copy_tensor(
                    X, batch_shape=warp_tf.batch_shape
                )
                expected_X_tf[..., indices] = k.cdf(
                    expected_X_tf[..., indices] * (1 - 2 * warp_tf._eps) + warp_tf._eps
                )

                self.assertAllClose(expected_X_tf, X_tf, **tols)

                # test untransform
                untransformed_X = warp_tf.untransform(X_tf)
                self.assertTrue(
                    torch.allclose(
                        untransformed_X,
                        expand_and_copy_tensor(X, batch_shape=warp_tf.batch_shape),
                        rtol=1e-3,
                        atol=1e-3 if self.device == torch.device("cpu") else 1e-2,
                    )
                )
                if len(warp_batch_shape) > 0:
                    with self.assertRaises(BotorchTensorDimensionError):
                        warp_tf.untransform(X_tf.unsqueeze(-3))

                # test no transform on eval
                warp_tf = get_test_warp(
                    d=3,
                    indices=indices,
                    transform_on_eval=False,
                    batch_shape=warp_batch_shape,
                    eps=eps,
                ).to(**tkwargs)
                X_tf = warp_tf(X)
                self.assertFalse(torch.equal(X, X_tf))
                warp_tf.eval()
                X_tf = warp_tf(X)
                self.assertTrue(torch.equal(X, X_tf))

                # test no transform on train
                warp_tf = get_test_warp(
                    d=3,
                    indices=indices,
                    transform_on_train=False,
                    batch_shape=warp_batch_shape,
                    eps=eps,
                ).to(**tkwargs)
                X_tf = warp_tf(X)
                self.assertTrue(torch.equal(X, X_tf))
                warp_tf.eval()
                X_tf = warp_tf(X)
                self.assertFalse(torch.equal(X, X_tf))

                # test equals
                warp_tf2 = get_test_warp(
                    d=3,
                    indices=indices,
                    transform_on_train=False,
                    batch_shape=warp_batch_shape,
                    eps=eps,
                ).to(**tkwargs)
                self.assertTrue(warp_tf.equals(warp_tf2))
                # test different transform_on_train
                warp_tf2 = get_test_warp(
                    d=3, indices=indices, batch_shape=warp_batch_shape, eps=eps
                )
                self.assertFalse(warp_tf.equals(warp_tf2))
                # test different indices
                warp_tf2 = get_test_warp(
                    d=3,
                    indices=[0, 1],
                    transform_on_train=False,
                    batch_shape=warp_batch_shape,
                    eps=eps,
                ).to(**tkwargs)
                self.assertFalse(warp_tf.equals(warp_tf2))

                # test preprocess_transform
                warp_tf.transform_on_train = False
                self.assertTrue(torch.equal(warp_tf.preprocess_transform(X), X))
                warp_tf.transform_on_train = True
                self.assertTrue(torch.equal(warp_tf.preprocess_transform(X), X_tf))

                # test _set_concentration
                warp_tf._set_concentration(0, warp_tf.concentration0)
                warp_tf._set_concentration(1, warp_tf.concentration1)

                # test concentration prior
                prior0 = LogNormalPrior(0.0, 0.75).to(**tkwargs)
                prior1 = LogNormalPrior(0.0, 0.5).to(**tkwargs)
                warp_tf = get_test_warp(
                    d=3,
                    indices=[0, 1],
                    concentration0_prior=prior0,
                    concentration1_prior=prior1,
                    batch_shape=warp_batch_shape,
                    eps=eps,
                )
                for i, (name, _, p, _, _) in enumerate(warp_tf.named_priors()):
                    self.assertEqual(name, f"concentration{i}_prior")
                    self.assertIsInstance(p, LogNormalPrior)
                    self.assertEqual(p.base_dist.scale, 0.75 if i == 0 else 0.5)

                # test non-unit cube bounds
                warp_tf = get_test_warp(
                    d=3,
                    indices=[0, 2],
                    eps=eps,
                    batch_shape=warp_batch_shape,
                    bounds=torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], **tkwargs),
                ).to(**tkwargs)
                X[..., indices] += 1
                X_tf = warp_tf(X)
                self.assertAllClose(expected_X_tf, X_tf, **tols)

            # test gradients
            X = 1 + 5 * torch.rand(*batch_shape, 4, 3, **tkwargs)
            X = X.unsqueeze(-3) if len(warp_batch_shape) > 0 else X
            warp_tf = get_test_warp(
                d=3, indices=indices, batch_shape=warp_batch_shape, eps=eps
            ).to(**tkwargs)
            X_tf = warp_tf(X)
            X_tf.sum().backward()
            for grad in (warp_tf.concentration0.grad, warp_tf.concentration1.grad):
                self.assertIsNotNone(grad)
                self.assertFalse(torch.isnan(grad).any())
                self.assertFalse(torch.isinf(grad).any())
                self.assertFalse((grad == 0).all())

            # test set with scalar
            warp_tf._set_concentration(i=0, value=2.0)
            self.assertTrue((warp_tf.concentration0 == 2.0).all())
            warp_tf._set_concentration(i=1, value=3.0)
            self.assertTrue((warp_tf.concentration1 == 3.0).all())

    def test_warp_mro(self) -> None:
        self.assertEqual(
            Warp.__mro__,
            (
                Warp,
                ReversibleInputTransform,
                InputTransform,
                GPyTorchModule,
                Module,
                ABC,
                object,
            ),
        )

    def test_numeric_to_categorical_encoding(self) -> None:
        # test exceptions
        with self.assertRaises(
            ValueError,
            msg="The number of categorical features exceeds the provided dimension.",
        ):
            categorical_features = {0: 2, 1: 3}
            NumericToCategoricalEncoding(
                dim=1,
                categorical_features=categorical_features,
                encoders={
                    0: partial(one_hot, num_classes=2),
                    1: partial(one_hot, num_classes=3),
                },
            )
        with self.assertRaises(
            ValueError,
            msg="The keys of `encoders` must match the keys of `categorical_features`.",
        ):
            categorical_features = {0: 2, 1: 3}
            NumericToCategoricalEncoding(
                dim=4,
                categorical_features=categorical_features,
                encoders={
                    0: partial(one_hot, num_classes=2),
                    2: partial(one_hot, num_classes=3),
                },
            )
        with self.assertRaises(
            ValueError,
            msg="The number of categorical features must be at least 1.",
        ):
            NumericToCategoricalEncoding(
                dim=2,
                categorical_features={},
                encoders={},
            )
        with self.assertRaises(
            ValueError,
            msg="Categorical feature at index 1",
        ):
            NumericToCategoricalEncoding(
                dim=2,
                categorical_features={1: 1},
                encoders={1: partial(one_hot, num_classes=1)},
            )
        with self.assertRaises(
            ValueError,
            msg="Categorical feature at index 1",
        ):
            NumericToCategoricalEncoding(
                dim=2,
                categorical_features={1: 2.0},
                encoders={1: partial(one_hot, num_classes=2)},
            )

        torch.manual_seed(42)
        for dtype in (torch.float, torch.double):
            # one categorical at start
            dim = 3
            categorical_features = {0: 3}
            tf = NumericToCategoricalEncoding(
                dim=dim,
                categorical_features=categorical_features,
                encoders={0: partial(one_hot, num_classes=3)},
            )
            tf.eval()
            cat_numeric = torch.randint(0, 3, (3,), device=self.device)
            cat_one_hot = one_hot(cat_numeric, num_classes=3)
            cont = torch.rand(3, 2, dtype=dtype, device=self.device)

            X_numeric = torch.cat(
                [cat_numeric.view(-1, 1), cont],
                dim=-1,
            ).to(dtype=dtype)

            expected = torch.cat(
                [cat_one_hot, cont],
                dim=-1,
            )
            X_one_hot = tf(X_numeric)
            self.assertTrue(torch.equal(X_one_hot, expected))

            # two categoricals at end
            dim = 4
            categorical_features = {2: 3, 3: 2}
            tf = NumericToCategoricalEncoding(
                dim=dim,
                categorical_features=categorical_features,
                encoders={
                    2: partial(one_hot, num_classes=3),
                    3: partial(one_hot, num_classes=2),
                },
            )
            tf.eval()
            cat_numeric1 = torch.randint(0, 3, (3,), device=self.device)
            cat_one_hot1 = one_hot(cat_numeric1, num_classes=3)
            cat_numeric2 = torch.randint(0, 2, (3,), device=self.device)
            cat_one_hot2 = one_hot(cat_numeric2, num_classes=2)
            cont = torch.rand(3, 2, dtype=dtype, device=self.device)

            X_numeric = torch.cat(
                [
                    cont,
                    cat_numeric1.view(-1, 1),
                    cat_numeric2.view(-1, 1),
                ],
                dim=-1,
            ).to(dtype=dtype)

            expected = torch.cat(
                [cont, cat_one_hot1, cat_one_hot2],
                dim=-1,
            )
            X_one_hot = tf(X_numeric)
            self.assertTrue(torch.equal(X_one_hot, expected))

            # two categoricals, one at start, one at end
            dim = 4
            categorical_features = {0: 3, 3: 2}
            tf = NumericToCategoricalEncoding(
                dim=dim,
                categorical_features=categorical_features,
                encoders={
                    0: partial(one_hot, num_classes=3),
                    3: partial(one_hot, num_classes=2),
                },
            )
            tf.eval()
            cat_numeric1 = torch.randint(0, 3, (3,), device=self.device)
            cat_one_hot1 = one_hot(cat_numeric1, num_classes=3)
            cat_numeric2 = torch.randint(0, 2, (3,), device=self.device)
            cat_one_hot2 = one_hot(cat_numeric2, num_classes=2)
            cont = torch.rand(3, 2, dtype=dtype, device=self.device)

            X_numeric = torch.cat(
                [
                    cat_numeric1.view(-1, 1).to(dtype=dtype, device=self.device),
                    cont,
                    cat_numeric2.view(-1, 1).to(dtype=dtype, device=self.device),
                ],
                dim=-1,
            )

            expected = torch.cat(
                [cat_one_hot1, cont, cat_one_hot2],
                dim=-1,
            )
            X_one_hot = tf(X_numeric)
            self.assertTrue(torch.equal(X_one_hot, expected))

            # only categoricals
            dim = 2
            categorical_features = {0: 3, 1: 2}
            tf = NumericToCategoricalEncoding(
                dim=dim,
                categorical_features=categorical_features,
                encoders={
                    0: partial(one_hot, num_classes=3),
                    1: partial(one_hot, num_classes=2),
                },
            )
            tf.eval()
            cat_numeric1 = torch.randint(0, 3, (3,), device=self.device)
            cat_one_hot1 = one_hot(cat_numeric1, num_classes=3)
            cat_numeric2 = torch.randint(0, 2, (3,), device=self.device)
            cat_one_hot2 = one_hot(cat_numeric2, num_classes=2)

            X_numeric = torch.cat(
                [
                    cat_numeric1.view(-1, 1).to(dtype=dtype, device=self.device),
                    cat_numeric2.view(-1, 1).to(dtype=dtype, device=self.device),
                ],
                dim=-1,
            )

            expected = torch.cat(
                [cat_one_hot1, cat_one_hot2],
                dim=-1,
            )
            X_one_hot = tf(X_numeric)
            self.assertTrue(torch.equal(X_one_hot, expected))

        # test no transform on eval
        tf = NumericToCategoricalEncoding(
            dim=dim,
            categorical_features=categorical_features,
            encoders={
                0: partial(one_hot, num_classes=3),
                1: partial(one_hot, num_classes=2),
            },
            transform_on_eval=False,
        )
        tf.eval()
        X_tf = tf(X_numeric)
        self.assertTrue(torch.equal(X_numeric, X_tf))

        # test no transform on train
        tf = NumericToCategoricalEncoding(
            dim=dim,
            categorical_features=categorical_features,
            encoders={
                0: partial(one_hot, num_classes=3),
                1: partial(one_hot, num_classes=2),
            },
            transform_on_train=False,
        )
        X_tf = tf(X_numeric)
        self.assertTrue(torch.equal(X_numeric, X_tf))
        tf.eval()
        X_tf = tf(X_numeric)
        self.assertFalse(torch.equal(X_numeric, X_tf))

        # test equals
        tf2 = NumericToCategoricalEncoding(
            dim=dim,
            categorical_features=categorical_features,
            encoders={
                0: partial(one_hot, num_classes=3),
                1: partial(one_hot, num_classes=2),
            },
            transform_on_train=False,
        )
        self.assertTrue(tf.equals(tf2))

        # test different transform_on_train
        tf3 = NumericToCategoricalEncoding(
            dim=dim,
            categorical_features=categorical_features,
            encoders={
                0: partial(one_hot, num_classes=3),
                1: partial(one_hot, num_classes=2),
            },
            transform_on_train=True,
        )
        self.assertFalse(tf3.equals(tf2))

        # test categorical features
        tf4 = NumericToCategoricalEncoding(
            dim=dim,
            categorical_features={0: 3},
            encoders={
                0: partial(one_hot, num_classes=3),
            },
            transform_on_train=True,
        )
        self.assertFalse(tf4.equals(tf3))

    def test_one_hot_to_numeric(self) -> None:
        dim = 8
        # test exceptions
        categorical_features = {0: 2, 1: 3}
        with self.assertRaises(ValueError, msg="Categorical features overlap."):
            OneHotToNumeric(dim=dim, categorical_features=categorical_features)
        dim = 3
        categorical_features = {0: 6}
        with self.assertRaises(
            ValueError, msg="Categorical features exceed the provided dimension."
        ):
            OneHotToNumeric(dim=dim, categorical_features=categorical_features)

        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 1000))
        for dtype in (torch.float, torch.double):
            # only categoricals
            dim = 5
            categorical_features = {0: 3, 3: 2}
            tf = OneHotToNumeric(dim=dim, categorical_features=categorical_features)
            tf.eval()
            self.assertEqual(tf.categorical_features, {0: 3, 3: 2})
            cat1_numeric = torch.randint(0, 3, (3,), device=self.device)
            cat1 = one_hot(cat1_numeric, num_classes=3)
            cat2_numeric = torch.randint(0, 2, (3,), device=self.device)
            cat2 = one_hot(cat2_numeric, num_classes=2)
            X = torch.cat([cat1, cat2], dim=-1)
            # test forward
            X_numeric = tf(X)
            expected = torch.cat(
                [
                    cat1_numeric.view(-1, 1).to(dtype=dtype, device=self.device),
                    cat2_numeric.view(-1, 1).to(dtype=dtype, device=self.device),
                ],
                dim=-1,
            )
            self.assertTrue(torch.equal(X_numeric, expected))
            X2 = tf.untransform(X_numeric)
            self.assertTrue(torch.equal(X2, X))

            # two categoricals at end
            dim = 8
            categorical_features = {6: 2, 3: 3}
            tf = OneHotToNumeric(dim=dim, categorical_features=categorical_features)
            tf.eval()
            self.assertEqual(tf.categorical_features, {3: 3, 6: 2})
            cat1_numeric = torch.randint(0, 3, (3,), device=self.device)
            cat1 = one_hot(cat1_numeric, num_classes=3)
            cat2_numeric = torch.randint(0, 2, (3,), device=self.device)
            cat2 = one_hot(cat2_numeric, num_classes=2)
            cont = torch.rand(3, 3, dtype=dtype, device=self.device)
            X = torch.cat([cont, cat1, cat2], dim=-1)
            # test forward
            X_numeric = tf(X)
            expected = torch.cat(
                [
                    cont,
                    cat1_numeric.view(-1, 1).to(cont),
                    cat2_numeric.view(-1, 1).to(cont),
                ],
                dim=-1,
            )
            self.assertTrue(torch.equal(X_numeric, expected))
            X2 = tf.untransform(X_numeric)
            self.assertTrue(torch.equal(X2, X))

            # three categoricals at end
            dim = 10
            categorical_features = {
                6: 2,
                3: 3,
                8: 2,
            }
            tf = OneHotToNumeric(dim=dim, categorical_features=categorical_features)
            tf.eval()
            self.assertEqual(tf.categorical_features, {3: 3, 6: 2, 8: 2})
            cat1_numeric = torch.randint(0, 3, (3,), device=self.device)
            cat1 = one_hot(cat1_numeric, num_classes=3)
            cat2_numeric = torch.randint(0, 2, (3,), device=self.device)
            cat2 = one_hot(cat2_numeric, num_classes=2)
            cat3_numeric = torch.randint(0, 2, (3,), device=self.device)
            cat3 = one_hot(cat3_numeric, num_classes=2)
            cont = torch.rand(3, 3, dtype=dtype, device=self.device)
            X = torch.cat([cont, cat1, cat2, cat3], dim=-1)
            # test forward
            X_numeric = tf(X)
            expected = torch.cat(
                [
                    cont,
                    cat1_numeric.view(-1, 1).to(cont),
                    cat2_numeric.view(-1, 1).to(cont),
                    cat3_numeric.view(-1, 1).to(cont),
                ],
                dim=-1,
            )
            self.assertTrue(torch.equal(X_numeric, expected))
            X2 = tf.untransform(X_numeric)
            self.assertTrue(torch.equal(X2, X))

            # three categoricals, one at start, two at end
            dim = 10
            categorical_features = {
                0: 3,
                6: 2,
                8: 2,
            }
            tf = OneHotToNumeric(dim=dim, categorical_features=categorical_features)
            tf.eval()
            self.assertEqual(tf.categorical_features, {0: 3, 6: 2, 8: 2})
            cat1_numeric = torch.randint(0, 3, (3,), device=self.device)
            cat1 = one_hot(cat1_numeric, num_classes=3)
            cat2_numeric = torch.randint(0, 2, (3,), device=self.device)
            cat2 = one_hot(cat2_numeric, num_classes=2)
            cat3_numeric = torch.randint(0, 2, (3,), device=self.device)
            cat3 = one_hot(cat3_numeric, num_classes=2)
            cont = torch.rand(3, 3, dtype=dtype, device=self.device)
            X = torch.cat([cat1, cont, cat2, cat3], dim=-1)
            # test forward
            X_numeric = tf(X)
            expected = torch.cat(
                [
                    cat1_numeric.view(-1, 1).to(cont),
                    cont,
                    cat2_numeric.view(-1, 1).to(cont),
                    cat3_numeric.view(-1, 1).to(cont),
                ],
                dim=-1,
            )
            self.assertTrue(torch.equal(X_numeric, expected))
            X2 = tf.untransform(X_numeric)
            self.assertTrue(torch.equal(X2, X))

            # test no categorical features.
            tf = OneHotToNumeric(dim=dim, categorical_features={})
            tf.eval()
            X_tf = tf(X)
            self.assertTrue(torch.equal(X, X_tf))
            X2 = tf.untransform(X_tf)
            self.assertTrue(torch.equal(X2, X_tf))

        # test no transform on eval
        tf2 = OneHotToNumeric(
            dim=dim, categorical_features=categorical_features, transform_on_eval=False
        )
        tf2.eval()
        X_tf = tf2(X)
        self.assertTrue(torch.equal(X, X_tf))

        # test no transform on train
        tf2 = OneHotToNumeric(
            dim=dim, categorical_features=categorical_features, transform_on_train=False
        )
        X_tf = tf2(X)
        self.assertTrue(torch.equal(X, X_tf))
        tf2.eval()
        X_tf = tf2(X)
        self.assertFalse(torch.equal(X, X_tf))

        # test equals
        tf3 = OneHotToNumeric(
            dim=dim, categorical_features=categorical_features, transform_on_train=False
        )
        self.assertTrue(tf3.equals(tf2))
        # test different transform_on_train
        tf3 = OneHotToNumeric(
            dim=dim, categorical_features=categorical_features, transform_on_train=True
        )
        self.assertFalse(tf3.equals(tf2))
        # test categorical features
        tf3 = OneHotToNumeric(
            dim=dim, categorical_features={}, transform_on_train=False
        )
        self.assertFalse(tf3.equals(tf2))


class TestAppendFeatures(BotorchTestCase):
    def test_append_features(self) -> None:
        with self.assertRaises(ValueError):
            AppendFeatures(torch.ones(1))
        with self.assertRaises(ValueError):
            AppendFeatures(torch.ones(3, 4, 2))

        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 100))

        for dtype in (torch.float, torch.double):
            feature_set = (
                torch.linspace(0, 1, 6).view(3, 2).to(device=self.device, dtype=dtype)
            )
            transform = AppendFeatures(feature_set=feature_set)
            self.assertTrue(transform.is_one_to_many)
            X = torch.rand(4, 5, 3, device=self.device, dtype=dtype)
            # in train - no transform
            transform.train()
            transformed_X = transform(X)
            self.assertTrue(torch.equal(X, transformed_X))
            # in eval - yes transform
            transform.eval()
            transformed_X = transform(X)
            self.assertFalse(torch.equal(X, transformed_X))
            self.assertEqual(transformed_X.shape, torch.Size([4, 15, 5]))
            self.assertTrue(
                torch.equal(transformed_X[..., :3], X.repeat_interleave(3, dim=-2))
            )
            self.assertTrue(
                torch.equal(transformed_X[..., 3:], feature_set.repeat(4, 5, 1))
            )
            # in fantasize - no transform
            with fantasize():
                transformed_X = transform(X)
            self.assertTrue(torch.equal(X, transformed_X))

            # Make sure .to calls work.
            transform.to(device=torch.device("cpu"), dtype=torch.half)
            self.assertEqual(transform.feature_set.device.type, "cpu")
            self.assertEqual(transform.feature_set.dtype, torch.half)

    def test_w_skip_expand(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            feature_set = torch.tensor([[0.0], [1.0]], **tkwargs)
            append_tf = AppendFeatures(feature_set=feature_set, skip_expand=True).eval()
            perturbation_set = torch.tensor([[0.0, 0.5], [1.0, 1.5]], **tkwargs)
            pert_tf = InputPerturbation(perturbation_set=perturbation_set).eval()
            test_X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], **tkwargs)
            tf_X = append_tf(pert_tf(test_X))
            expected_X = torch.tensor(
                [
                    [0.0, 0.5, 0.0],
                    [1.0, 1.5, 1.0],
                    [1.0, 1.5, 0.0],
                    [2.0, 2.5, 1.0],
                ],
                **tkwargs,
            )
            self.assertAllClose(tf_X, expected_X)
            # Batched evaluation.
            tf_X = append_tf(pert_tf(test_X.expand(3, 5, -1, -1)))
            self.assertAllClose(tf_X, expected_X.expand(3, 5, -1, -1))

    def test_w_f(self) -> None:
        def f1(x: Tensor, n_f: int = 1) -> Tensor:
            result = torch.sum(x, dim=-1, keepdim=True).unsqueeze(-2)
            return result.expand(*result.shape[:-2], n_f, -1)

        def f2(x: Tensor, n_f: int = 1) -> Tensor:
            result = x[..., -2:].unsqueeze(-2)
            return result.expand(*result.shape[:-2], n_f, -1)

        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 100))

        for dtype in [torch.float, torch.double]:
            tkwargs = {"device": self.device, "dtype": dtype}

            # test init
            with self.assertRaises(ValueError):
                transform = AppendFeatures(f=f1, indices=[0, 0])
            with self.assertRaises(ValueError):
                transform = AppendFeatures(f=f1, indices=[])
            with self.assertRaises(ValueError):
                transform = AppendFeatures(f=f1, skip_expand=True)
            with self.assertRaises(ValueError):
                transform = AppendFeatures(feature_set=None, f=None)
            with self.assertRaises(ValueError):
                transform = AppendFeatures(
                    feature_set=torch.linspace(0, 1, 6)
                    .view(3, 2)
                    .to(device=self.device, dtype=dtype),
                    f=f1,
                )

            # test functionality with n_f = 1
            X = torch.rand(1, 3, **tkwargs)
            transform = AppendFeatures(
                f=f1,
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((1, 4)))
            self.assertAllClose(X.sum(dim=-1), X_transformed[..., -1])

            X = torch.rand(10, 3, **tkwargs)
            transform = AppendFeatures(
                f=f1,
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((10, 4)))
            self.assertAllClose(X.sum(dim=-1), X_transformed[..., -1])

            transform = AppendFeatures(
                f=f1,
                indices=[0, 1],
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((10, 4)))
            self.assertTrue(
                torch.allclose(X[..., [0, 1]].sum(dim=-1), X_transformed[..., -1])
            )

            transform = AppendFeatures(
                f=f2,
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((10, 5)))

            X = torch.rand(1, 10, 3).to(**tkwargs)
            transform = AppendFeatures(
                f=f1,
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((1, 10, 4)))

            X = torch.rand(1, 1, 3).to(**tkwargs)
            transform = AppendFeatures(
                f=f1,
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((1, 1, 4)))

            X = torch.rand(2, 10, 3).to(**tkwargs)
            transform = AppendFeatures(
                f=f1,
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((2, 10, 4)))

            transform = AppendFeatures(
                f=f2,
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((2, 10, 5)))
            self.assertAllClose(X[..., -2:], X_transformed[..., -2:])

            # test functionality with n_f > 1
            X = torch.rand(10, 3, **tkwargs)
            transform = AppendFeatures(
                f=f1,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((20, 4)))

            X = torch.rand(2, 10, 3, **tkwargs)
            transform = AppendFeatures(
                f=f1,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((2, 20, 4)))

            X = torch.rand(1, 10, 3, **tkwargs)
            transform = AppendFeatures(
                f=f1,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((1, 20, 4)))

            X = torch.rand(1, 3, **tkwargs)
            transform = AppendFeatures(
                f=f1,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((2, 4)))

            X = torch.rand(10, 3, **tkwargs)
            transform = AppendFeatures(
                f=f2,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((20, 5)))

            X = torch.rand(2, 10, 3, **tkwargs)
            transform = AppendFeatures(
                f=f2,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((2, 20, 5)))

            X = torch.rand(1, 10, 3, **tkwargs)
            transform = AppendFeatures(
                f=f2,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((1, 20, 5)))

            X = torch.rand(1, 3, **tkwargs)
            transform = AppendFeatures(
                f=f2,
                fkwargs={"n_f": 2},
                transform_on_eval=True,
                transform_on_train=True,
                transform_on_fantasize=True,
            )
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((2, 5)))

            # test no transform on train
            X = torch.rand(10, 3).to(**tkwargs)
            transform = AppendFeatures(
                f=f1, transform_on_train=False, transform_on_eval=True
            )
            transform.train()
            X_transformed = transform(X)
            self.assertTrue(torch.equal(X, X_transformed))
            transform.eval()
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((10, 4)))

            # test not transform on eval
            X = torch.rand(10, 3).to(**tkwargs)
            transform = AppendFeatures(
                f=f1, transform_on_eval=False, transform_on_train=True
            )
            transform.eval()
            X_transformed = transform(X)
            self.assertTrue(torch.equal(X, X_transformed))
            transform.train()
            X_transformed = transform(X)
            self.assertEqual(X_transformed.shape, torch.Size((10, 4)))


class TestInteractionFeatures(BotorchTestCase):
    def test_interaction_features(self) -> None:
        interaction = InteractionFeatures()
        X = torch.arange(6, dtype=torch.float).reshape(2, 3)
        X_tf = interaction(X)
        self.assertTrue(X_tf.shape, torch.Size([2, 6]))

        # test correct output values
        self.assertTrue(
            torch.equal(
                X_tf,
                torch.tensor(
                    [[0.0, 1.0, 2.0, 0.0, 0.0, 2.0], [3.0, 4.0, 5.0, 12.0, 15.0, 20.0]]
                ),
            )
        )
        X = torch.arange(6, dtype=torch.float).reshape(2, 3)
        interaction = InteractionFeatures(indices=[1, 2])
        X_tf = interaction(X)
        self.assertTrue(
            torch.equal(
                X_tf,
                torch.tensor([[0.0, 1.0, 2.0, 2.0], [3.0, 4.0, 5.0, 20.0]]),
            )
        )
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            interaction(torch.rand(4, 2))

        # test batched evaluation
        interaction = InteractionFeatures()
        X_tf = interaction(torch.rand(4, 2, 4))
        self.assertTrue(X_tf.shape, torch.Size([4, 2, 10]))

        X_tf = interaction(torch.rand(5, 7, 3, 4))
        self.assertTrue(X_tf.shape, torch.Size([5, 7, 3, 10]))


class TestFilterFeatures(BotorchTestCase):
    def test_filter_features(self) -> None:
        with self.assertRaises(ValueError):
            FilterFeatures(torch.tensor([[1, 2]], dtype=torch.long))
        with self.assertRaises(ValueError):
            FilterFeatures(torch.tensor([1.0, 2.0]))
        with self.assertRaises(ValueError):
            FilterFeatures(torch.tensor([-1, 0, 1], dtype=torch.long))
        with self.assertRaises(ValueError):
            FilterFeatures(torch.tensor([0, 1, 1], dtype=torch.long))

        # set seed to range where this is known to not be flaky
        torch.manual_seed(randint(0, 100))

        for dtype in (torch.float, torch.double):
            feature_indices = torch.tensor(
                [0, 2, 3, 5], dtype=torch.long, device=self.device
            )
            transform = FilterFeatures(feature_indices=feature_indices)
            X = torch.rand(4, 5, 6, device=self.device, dtype=dtype)
            # in train - yes transform
            transform.train()
            transformed_X = transform(X)
            self.assertFalse(torch.equal(X, transformed_X))
            self.assertEqual(transformed_X.shape, torch.Size([4, 5, 4]))
            self.assertTrue(torch.equal(transformed_X, X[..., feature_indices]))
            # in eval - yes transform
            transform.eval()
            transformed_X = transform(X)
            self.assertFalse(torch.equal(X, transformed_X))
            self.assertEqual(transformed_X.shape, torch.Size([4, 5, 4]))
            self.assertTrue(torch.equal(transformed_X, X[..., feature_indices]))
            # in fantasize - yes transform
            with fantasize():
                transformed_X = transform(X)
                self.assertFalse(torch.equal(X, transformed_X))
                self.assertEqual(transformed_X.shape, torch.Size([4, 5, 4]))
                self.assertTrue(torch.equal(transformed_X, X[..., feature_indices]))

            # Make sure .to calls work.
            transform.to(device=torch.device("cpu"))
            self.assertEqual(transform.feature_indices.device.type, "cpu")
            # test equals
            transform2 = FilterFeatures(feature_indices=feature_indices)
            self.assertTrue(transform.equals(transform2))
            # test different indices
            feature_indices2 = torch.tensor(
                [0, 2, 3, 6], dtype=torch.long, device=self.device
            )
            transform2 = FilterFeatures(feature_indices=feature_indices2)
            self.assertFalse(transform.equals(transform2))
            # test different length
            feature_indices2 = torch.tensor(
                [2, 3, 5], dtype=torch.long, device=self.device
            )
            transform2 = FilterFeatures(feature_indices=feature_indices2)
            self.assertFalse(transform.equals(transform2))
            # test different transform_on_train
            transform2 = FilterFeatures(
                feature_indices=feature_indices, transform_on_train=False
            )
            self.assertFalse(transform.equals(transform2))
            # test different transform_on_eval
            transform2 = FilterFeatures(
                feature_indices=feature_indices, transform_on_eval=False
            )
            self.assertFalse(transform.equals(transform2))
            # test different transform_on_fantasize
            transform2 = FilterFeatures(
                feature_indices=feature_indices, transform_on_fantasize=False
            )
            self.assertFalse(transform.equals(transform2))


class TestInputPerturbation(BotorchTestCase):
    def test_input_perturbation(self) -> None:
        with self.assertRaisesRegex(ValueError, "-dim tensor!"):
            InputPerturbation(torch.ones(1))
        with self.assertRaisesRegex(ValueError, "-dim tensor!"):
            InputPerturbation(torch.ones(3, 2, 1))
        with self.assertRaisesRegex(ValueError, "the same number of columns"):
            InputPerturbation(torch.ones(2, 1), bounds=torch.ones(2, 4))

        for dtype in (torch.float, torch.double):
            perturbation_set = torch.tensor(
                [[0.5, -0.3], [0.2, 0.4], [-0.7, 0.1]], device=self.device, dtype=dtype
            )
            transform = InputPerturbation(perturbation_set=perturbation_set)
            self.assertTrue(transform.is_one_to_many)
            X = torch.tensor(
                [[[0.5, 0.5], [0.9, 0.7]], [[0.3, 0.2], [0.1, 0.4]]],
                device=self.device,
                dtype=dtype,
            )
            expected = torch.tensor(
                [
                    [
                        [1.0, 0.2],
                        [0.7, 0.9],
                        [-0.2, 0.6],
                        [1.4, 0.4],
                        [1.1, 1.1],
                        [0.2, 0.8],
                    ],
                    [
                        [0.8, -0.1],
                        [0.5, 0.6],
                        [-0.4, 0.3],
                        [0.6, 0.1],
                        [0.3, 0.8],
                        [-0.6, 0.5],
                    ],
                ],
                device=self.device,
                dtype=dtype,
            )
            # in train - no transform
            transformed = transform(X)
            self.assertTrue(torch.equal(transformed, X))
            # in eval - transform
            transform.eval()
            transformed = transform(X)
            self.assertAllClose(transformed, expected)
            # in fantasize - no transform
            with fantasize():
                transformed = transform(X)
            self.assertTrue(torch.equal(transformed, X))
            # with bounds
            bounds = torch.tensor(
                [[0.0, 0.2], [1.2, 0.9]], device=self.device, dtype=dtype
            )
            transform = InputPerturbation(
                perturbation_set=perturbation_set, bounds=bounds
            )
            transform.eval()
            expected = torch.tensor(
                [
                    [
                        [1.0, 0.2],
                        [0.7, 0.9],
                        [0.0, 0.6],
                        [1.2, 0.4],
                        [1.1, 0.9],
                        [0.2, 0.8],
                    ],
                    [
                        [0.8, 0.2],
                        [0.5, 0.6],
                        [0.0, 0.3],
                        [0.6, 0.2],
                        [0.3, 0.8],
                        [0.0, 0.5],
                    ],
                ],
                device=self.device,
                dtype=dtype,
            )
            transformed = transform(X)
            self.assertAllClose(transformed, expected)

            # Make sure .to calls work.
            transform.to(device=torch.device("cpu"), dtype=torch.half)
            self.assertEqual(transform.perturbation_set.device.type, "cpu")
            self.assertEqual(transform.perturbation_set.dtype, torch.half)
            self.assertEqual(transform.bounds.device.type, "cpu")
            self.assertEqual(transform.bounds.dtype, torch.half)

            # multiplicative
            perturbation_set = torch.tensor(
                [[0.5, 1.5], [1.0, 2.0]],
                device=self.device,
                dtype=dtype,
            )
            transform = InputPerturbation(
                perturbation_set=perturbation_set, multiplicative=True
            )
            transform.eval()
            transformed = transform(X)
            expected = torch.tensor(
                [
                    [[0.25, 0.75], [0.5, 1.0], [0.45, 1.05], [0.9, 1.4]],
                    [[0.15, 0.3], [0.3, 0.4], [0.05, 0.6], [0.1, 0.8]],
                ],
                device=self.device,
                dtype=dtype,
            )
            self.assertAllClose(transformed, expected)

            # heteroscedastic
            def perturbation_generator(X: Tensor) -> Tensor:
                return torch.stack([X * 0.1, X * 0.2], dim=-2)

            transform = InputPerturbation(
                perturbation_set=perturbation_generator
            ).eval()
            transformed = transform(X)
            expected = torch.stack(
                [
                    X[..., 0, :] * 1.1,
                    X[..., 0, :] * 1.2,
                    X[..., 1, :] * 1.1,
                    X[..., 1, :] * 1.2,
                ],
                dim=-2,
            )
            self.assertAllClose(transformed, expected)

            # testing same heteroscedastic transform with subset of indices
            indices = [0, 1]
            subset_transform = InputPerturbation(
                perturbation_set=perturbation_generator, indices=indices
            ).eval()
            X_repeat = X.repeat(1, 1, 2)
            subset_transformed = subset_transform(X_repeat)
            # first set of two indices are the same as with previous transform
            self.assertAllClose(subset_transformed[..., :2], expected)

            # second set of two indices are untransformed but have expanded batch shape
            num_pert = subset_transform.batch_shape[-1]
            sec_expected = X.unsqueeze(-2).expand(*X.shape[:-1], num_pert, -1)
            sec_expected = sec_expected.flatten(-3, -2)
            self.assertAllClose(subset_transformed[..., 2:], sec_expected)
