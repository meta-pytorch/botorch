#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from unittest.mock import MagicMock, patch

from botorch.utils.testing import BotorchTestCase
from botorch_community.models.utils.prior_fitted_network import (
    ACCEPT_LICENSE_ENV_VAR,
    ensure_license_accepted,
    MODEL_LICENSES,
    ModelLicense,
    ModelPaths,
    save_license_copy,
)

NOTICE_LICENSE = ModelLicense(
    name="Apache-2.0",
    url="https://example.com/license",
    text_url="https://example.com/LICENSE.txt",
    requires_acceptance=False,
)
GATED_LICENSE = ModelLicense(
    name="Example Gated License",
    url="https://example.com/gated-license",
    text_url="https://example.com/gated/LICENSE.txt",
    requires_acceptance=True,
    attribution="Built with Example",
)


class TestModelLicenses(BotorchTestCase):
    def test_registry_covers_default_models(self):
        for model_path in ModelPaths:
            self.assertIn(model_path.value, MODEL_LICENSES)
        self.assertIn("Prior-Labs/TabPFN-v2-reg", MODEL_LICENSES)
        self.assertTrue(MODEL_LICENSES["Prior-Labs/TabPFN-v2-reg"].requires_acceptance)

    def test_notice_only_license_never_raises(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            ensure_license_accepted(NOTICE_LICENSE, cache_dir=cache_dir)
            # no acceptance marker is needed or written
            self.assertEqual([f for f in os.listdir(cache_dir) if "accepted" in f], [])

    def test_gated_license_raises_without_acceptance(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            env = {k: v for k, v in os.environ.items() if k != ACCEPT_LICENSE_ENV_VAR}
            with patch.dict(os.environ, env, clear=True):
                with patch("sys.stdin") as mock_stdin:
                    mock_stdin.isatty.return_value = False
                    with self.assertRaisesRegex(RuntimeError, "accept_license"):
                        ensure_license_accepted(GATED_LICENSE, cache_dir=cache_dir)

    def test_gated_license_accepted_via_kwarg_and_marker_persists(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch(
                "botorch_community.models.utils.prior_fitted_network."
                "save_license_copy"
            ) as mock_save:
                ensure_license_accepted(
                    GATED_LICENSE, accept_license=True, cache_dir=cache_dir
                )
                mock_save.assert_called_once()
            # acceptance was recorded: subsequent calls need no consent
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = False
                ensure_license_accepted(GATED_LICENSE, cache_dir=cache_dir)

    def test_gated_license_accepted_via_env_var(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.dict(os.environ, {ACCEPT_LICENSE_ENV_VAR: "1"}):
                with patch(
                    "botorch_community.models.utils.prior_fitted_network."
                    "save_license_copy"
                ):
                    ensure_license_accepted(GATED_LICENSE, cache_dir=cache_dir)

    def test_save_license_copy(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            response = MagicMock()
            response.text = "LICENSE TEXT"
            with patch(
                "botorch_community.models.utils.prior_fitted_network." "requests.get",
                return_value=response,
            ) as mock_get:
                path = save_license_copy(GATED_LICENSE, cache_dir=cache_dir)
                self.assertTrue(os.path.exists(path))
                with open(path) as f:
                    self.assertEqual(f.read(), "LICENSE TEXT")
                # cached: a second call does not re-download
                path2 = save_license_copy(GATED_LICENSE, cache_dir=cache_dir)
                self.assertEqual(path, path2)
                mock_get.assert_called_once()
            # licenses without a text_url are skipped
            no_text = ModelLicense(name="X", url="https://example.com/x")
            self.assertIsNone(save_license_copy(no_text, cache_dir=cache_dir))

    def test_download_model_checks_license(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch(
                "botorch_community.models.utils.prior_fitted_network."
                "ensure_license_accepted"
            ) as mock_ensure:
                import torch

                # the cached-file branch avoids any network access
                from botorch_community.models.utils.prior_fitted_network import (
                    download_model,
                )

                cache_path = os.path.join(
                    cache_dir, ModelPaths.pfns4bo_hebo.value.split("/")[-1]
                )
                torch.save(torch.nn.Linear(1, 1), cache_path)
                download_model(ModelPaths.pfns4bo_hebo, cache_dir=cache_dir)
                mock_ensure.assert_called_once()
