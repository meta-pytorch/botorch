#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch import settings
from botorch.exceptions import InputDataError, InputDataWarning
from botorch.models.utils import (
    add_output_dim,
    check_min_max_scaling,
    check_no_nans,
    check_standardization,
    fantasize,
    gpt_posterior_settings,
    multioutput_to_batch_mode_transform,
    validate_input_scaling,
)

from botorch.models.utils.assorted import consolidate_duplicates, detect_duplicates
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings as gpt_settings


class TestMultiOutputToBatchModeTransform(BotorchTestCase):
    def test_multioutput_to_batch_mode_transform(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            n = 3
            num_outputs = 2
            train_X = torch.rand(n, 1, **tkwargs)
            train_Y = torch.rand(n, num_outputs, **tkwargs)
            train_Yvar = torch.rand(n, num_outputs, **tkwargs)
            X_out, Y_out, Yvar_out = multioutput_to_batch_mode_transform(
                train_X=train_X,
                train_Y=train_Y,
                num_outputs=num_outputs,
                train_Yvar=train_Yvar,
            )
            expected_X_out = train_X.unsqueeze(0).expand(num_outputs, -1, 1)
            self.assertTrue(torch.equal(X_out, expected_X_out))
            self.assertTrue(torch.equal(Y_out, train_Y.transpose(0, 1)))
            self.assertTrue(torch.equal(Yvar_out, train_Yvar.transpose(0, 1)))


class TestAddOutputDim(BotorchTestCase):
    def test_add_output_dim(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            original_batch_shape = torch.Size([2])
            # check exception is raised when trailing batch dims do not line up
            X = torch.rand(2, 3, 2, 1, **tkwargs)
            with self.assertRaises(RuntimeError):
                add_output_dim(X=X, original_batch_shape=original_batch_shape)
            # test no new batch dims
            X = torch.rand(2, 2, 1, **tkwargs)
            X_out, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=original_batch_shape
            )
            self.assertTrue(torch.equal(X_out, X.unsqueeze(1)))
            self.assertEqual(output_dim_idx, 1)
            # test new batch dims
            X = torch.rand(3, 2, 2, 1, **tkwargs)
            X_out, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=original_batch_shape
            )
            self.assertTrue(torch.equal(X_out, X.unsqueeze(2)))
            self.assertEqual(output_dim_idx, 2)


class TestInputDataChecks(BotorchTestCase):
    def setUp(self) -> None:
        # The super class usually disables input data warnings in unit tests.
        # Don't do that here.
        super().setUp(suppress_input_warnings=False)

    def test_check_no_nans(self):
        check_no_nans(torch.tensor([1.0, 2.0]))
        with self.assertRaises(InputDataError):
            check_no_nans(torch.tensor([1.0, float("nan")]))

    def test_check_min_max_scaling(self):
        # check unscaled input in unit cube
        X = 0.1 + 0.8 * torch.rand(4, 2, 3)
        with warnings.catch_warnings(record=True) as ws:
            check_min_max_scaling(X=X)
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        check_min_max_scaling(X=X, raise_on_fail=True)
        with self.assertWarnsRegex(
            expected_warning=InputDataWarning, expected_regex="not scaled"
        ):
            check_min_max_scaling(X=X, strict=True)
        with self.assertRaises(InputDataError):
            check_min_max_scaling(X=X, strict=True, raise_on_fail=True)
        # check proper input
        Xmin, Xmax = X.min(dim=-1, keepdim=True)[0], X.max(dim=-1, keepdim=True)[0]
        Xstd = (X - Xmin) / (Xmax - Xmin)
        with warnings.catch_warnings(record=True) as ws:
            check_min_max_scaling(X=Xstd)
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        check_min_max_scaling(X=Xstd, raise_on_fail=True)
        with warnings.catch_warnings(record=True) as ws:
            check_min_max_scaling(X=Xstd, strict=True)
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        check_min_max_scaling(X=Xstd, strict=True, raise_on_fail=True)
        # check violation
        X[0, 0, 0] = 2
        with warnings.catch_warnings(record=True) as ws:
            check_min_max_scaling(X=X)
        self.assertTrue(any(issubclass(w.category, InputDataWarning) for w in ws))
        self.assertTrue(any("not contained" in str(w.message) for w in ws))
        with self.assertRaises(InputDataError):
            check_min_max_scaling(X=X, raise_on_fail=True)
        with warnings.catch_warnings(record=True) as ws:
            check_min_max_scaling(X=X, strict=True)
        self.assertTrue(any(issubclass(w.category, InputDataWarning) for w in ws))
        self.assertTrue(any("not contained" in str(w.message) for w in ws))
        with self.assertRaises(InputDataError):
            check_min_max_scaling(X=X, strict=True, raise_on_fail=True)
        # check ignore_dims
        with warnings.catch_warnings(record=True) as ws:
            check_min_max_scaling(X=X, ignore_dims=[0])
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        # all dims ignored
        with warnings.catch_warnings(record=True) as ws:
            check_min_max_scaling(X=X, ignore_dims=[0, 1, 2])
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))

    def test_check_standardization(self):
        # Ensure that it is not filtered out.
        warnings.filterwarnings("always", category=InputDataWarning)
        torch.manual_seed(0)
        Y = torch.randn(3, 4, 2)
        # check standardized input
        Yst = (Y - Y.mean(dim=-2, keepdim=True)) / Y.std(dim=-2, keepdim=True)
        with warnings.catch_warnings(record=True) as ws:
            check_standardization(Y=Yst)
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        check_standardization(Y=Yst, raise_on_fail=True)

        # check standardized input with one observation
        y = torch.zeros((3, 1, 2))
        with warnings.catch_warnings(record=True) as ws:
            check_standardization(Y=y)
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        check_standardization(Y=y, raise_on_fail=True)

        # check nonzero mean for case where >= 2 observations per batch
        msg_more_than_1_obs = r"Data \(outcome observations\) is not standardized \(std"
        with self.assertWarnsRegex(InputDataWarning, msg_more_than_1_obs):
            check_standardization(Y=Yst + 1)
        with self.assertRaisesRegex(InputDataError, msg_more_than_1_obs):
            check_standardization(Y=Yst + 1, raise_on_fail=True)

        # check nonzero mean for case where < 2 observations per batch
        msg_one_obs = r"Data \(outcome observations\) is not standardized \(mean ="
        y = torch.ones((3, 1, 2), dtype=torch.float32)
        with self.assertWarnsRegex(InputDataWarning, msg_one_obs):
            check_standardization(Y=y)
        with self.assertRaisesRegex(InputDataError, msg_one_obs):
            check_standardization(Y=y, raise_on_fail=True)

        # check non-unit variance
        with self.assertWarnsRegex(InputDataWarning, msg_more_than_1_obs):
            check_standardization(Y=Yst * 2)
        with self.assertRaisesRegex(InputDataError, msg_more_than_1_obs):
            check_standardization(Y=Yst * 2, raise_on_fail=True)

    def test_validate_input_scaling(self):
        train_X = 2 + torch.rand(3, 4, 3)
        train_Y = torch.randn(3, 4, 2)
        # check that nothing is being checked
        with settings.validate_input_scaling(False), warnings.catch_warnings(
            record=True
        ) as ws:
            validate_input_scaling(train_X=train_X, train_Y=train_Y)
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        # check that warnings are being issued
        with warnings.catch_warnings(record=True) as ws:
            validate_input_scaling(train_X=train_X, train_Y=train_Y)
        self.assertTrue(any(issubclass(w.category, InputDataWarning) for w in ws))
        # check that errors are raised when requested
        with self.assertRaises(InputDataError):
            validate_input_scaling(train_X=train_X, train_Y=train_Y, raise_on_fail=True)
        # check that normalization & standardization checks & errors are skipped when
        # check_nans_only is True
        validate_input_scaling(
            train_X=train_X, train_Y=train_Y, raise_on_fail=True, check_nans_only=True
        )
        # check that no errors are being raised if everything is standardized
        train_X_min = train_X.min(dim=-1, keepdim=True)[0]
        train_X_max = train_X.max(dim=-1, keepdim=True)[0]
        train_X_std = (train_X - train_X_min) / (train_X_max - train_X_min)
        train_Y_std = (train_Y - train_Y.mean(dim=-2, keepdim=True)) / train_Y.std(
            dim=-2, keepdim=True
        )
        with warnings.catch_warnings(record=True) as ws:
            validate_input_scaling(train_X=train_X_std, train_Y=train_Y_std)
        self.assertFalse(any(issubclass(w.category, InputDataWarning) for w in ws))
        # test that negative variances raise an error
        train_Yvar = torch.rand_like(train_Y_std)
        train_Yvar[0, 0, 1] = -0.5
        with self.assertRaises(InputDataError):
            validate_input_scaling(
                train_X=train_X_std, train_Y=train_Y_std, train_Yvar=train_Yvar
            )
        # check that NaNs raise errors
        train_X_std[0, 0, 0] = float("nan")
        with self.assertRaises(InputDataError):
            validate_input_scaling(train_X=train_X_std, train_Y=train_Y_std)
        # NaNs still raise errors when check_nans_only is True
        with self.assertRaises(InputDataError):
            validate_input_scaling(
                train_X=train_X_std, train_Y=train_Y_std, check_nans_only=True
            )


class TestGPTPosteriorSettings(BotorchTestCase):
    def test_gpt_posterior_settings(self):
        for propagate_grads in (False, True):
            with settings.propagate_grads(propagate_grads):
                with gpt_posterior_settings():
                    self.assertTrue(gpt_settings.debug.off())
                    self.assertTrue(gpt_settings.fast_pred_var.on())
                    if settings.propagate_grads.off():
                        self.assertTrue(gpt_settings.detach_test_caches.on())
                    else:
                        self.assertTrue(gpt_settings.detach_test_caches.off())


class TestFantasize(BotorchTestCase):
    def test_fantasize(self):
        self.assertFalse(fantasize.on())
        self.assertTrue(fantasize.off())
        with fantasize():
            self.assertTrue(fantasize.on())
            self.assertFalse(fantasize.off())
        with fantasize(False):
            self.assertFalse(fantasize.on())
            self.assertTrue(fantasize.off())


class TestConsolidation(BotorchTestCase):
    def test_consolidation(self):
        X = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
            ]
        )
        Y = torch.tensor([[0, 1], [2, 3]])
        expected_X = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
            ]
        )
        expected_Y = torch.tensor([[0, 1], [0, 2]])
        expected_new_indices = torch.tensor([0, 1, 0, 2])

        # deduped case
        consolidated_X, consolidated_Y, new_indices = consolidate_duplicates(X=X, Y=Y)
        self.assertTrue(torch.equal(consolidated_X, expected_X))
        self.assertTrue(torch.equal(consolidated_Y, expected_Y))
        self.assertTrue(torch.equal(new_indices, expected_new_indices))

        # test rtol
        big_X = torch.tensor(
            [
                [10000.0, 20000.0, 30000.0],
                [20000.0, 30000.0, 40000.0],
                [10000.0, 20000.0, 30001.0],
                [30000.0, 40000.0, 50000.0],
            ]
        )
        expected_big_X = torch.tensor(
            [
                [10000.0, 20000.0, 30000.0],
                [20000.0, 30000.0, 40000.0],
                [30000.0, 40000.0, 50000.0],
            ]
        )
        # rtol is not used by default
        consolidated_X, consolidated_Y, new_indices = consolidate_duplicates(
            X=big_X, Y=Y
        )
        self.assertTrue(torch.equal(consolidated_X, big_X))
        self.assertTrue(torch.equal(consolidated_Y, Y))
        self.assertTrue(torch.equal(new_indices, torch.tensor([0, 1, 2, 3])))
        # when rtol is used
        consolidated_X, consolidated_Y, new_indices = consolidate_duplicates(
            X=big_X, Y=Y, rtol=1e-4, atol=0
        )
        self.assertTrue(torch.equal(consolidated_X, expected_big_X))
        self.assertTrue(torch.equal(consolidated_Y, expected_Y))
        self.assertTrue(torch.equal(new_indices, expected_new_indices))

        # not deduped case
        no_dup_X = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ]
        )
        consolidated_X, consolidated_Y, new_indices = consolidate_duplicates(
            X=no_dup_X, Y=Y
        )
        self.assertTrue(torch.equal(consolidated_X, no_dup_X))
        self.assertTrue(torch.equal(consolidated_Y, Y))
        self.assertTrue(torch.equal(new_indices, torch.tensor([0, 1, 2, 3])))

        # test batch shape
        with self.assertRaises(ValueError):
            consolidate_duplicates(X=X.repeat(2, 1, 1), Y=Y.repeat(2, 1, 1))

        with self.assertRaises(ValueError):
            detect_duplicates(X=X.repeat(2, 1, 1))

        # test chain link edge case
        close_X = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.4],
                [1.0, 2.0, 3.8],
                [1.0, 2.0, 4.2],
            ]
        )
        consolidated_X, consolidated_Y, new_indices = consolidate_duplicates(
            X=close_X, Y=Y, rtol=0, atol=0.5
        )
        self.assertTrue(torch.equal(consolidated_X, close_X))
        self.assertTrue(torch.equal(consolidated_Y, Y))
        self.assertTrue(torch.equal(new_indices, torch.tensor([0, 1, 2, 3])))
