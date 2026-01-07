#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from botorch.exceptions import UnsupportedError

from botorch.utils.test_helpers import get_pvar_expected
from botorch.utils.testing import BotorchTestCase


class TestTestHelpers(BotorchTestCase):
    def test_get_pvar_expected(self):
        # The test helper is used throughout, veryfying that an error is raised
        # when it is used with an unsupported model.
        with self.assertRaisesRegex(
            UnsupportedError,
            "`get_pvar_expected` only supports `BatchedMultiOutputGPyTorchModel`s.",
        ):
            get_pvar_expected(
                posterior=mock.Mock(), model=mock.Mock(), X=mock.Mock(), m=2
            )
