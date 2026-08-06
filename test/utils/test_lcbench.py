#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow
import torch
from botorch.utils import lcbench
from botorch.utils.lcbench import (
    _read_lcbench_parquet,
    DEFAULT_LCBENCH_METRIC_NAME,
    LCBENCH_DATASET_NAMES,
    LCBENCH_LOG_SCALE_PARAMETER_NAMES,
    LCBENCH_PARAMETER_NAMES,
    load_lcbench_data,
)
from botorch.utils.testing import BotorchTestCase

N_TRIALS = 3
N_RECORDED_EPOCHS = 52


def _make_config_df() -> pd.DataFrame:
    """A config frame with the 7 parameters (shuffled) plus an extra column."""
    rng = np.random.default_rng(0)
    columns = {
        name: rng.uniform(1.0, 2.0, size=N_TRIALS)
        for name in reversed(LCBENCH_PARAMETER_NAMES)
    }
    columns["OpenML_task_id"] = np.arange(N_TRIALS, dtype=float)
    return pd.DataFrame(columns, index=pd.RangeIndex(N_TRIALS, name="trial"))


def _make_metrics_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [range(N_TRIALS), range(N_RECORDED_EPOCHS)], names=["trial", "epoch"]
    )
    values = np.arange(N_TRIALS * N_RECORDED_EPOCHS, dtype=float)
    return pd.DataFrame(
        {DEFAULT_LCBENCH_METRIC_NAME: values, "time": values}, index=index
    )


class TestLCBench(BotorchTestCase):
    def test_missing_pandas_raises(self) -> None:
        with patch.object(lcbench, "_HAS_PANDAS", False):
            with self.assertRaisesRegex(ImportError, "botorch\\[lcbench\\]"):
                load_lcbench_data("Fashion-MNIST")

    def test_invalid_dataset_name_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid dataset 'not_a_dataset'"):
            load_lcbench_data("not_a_dataset")

    def test_read_parquet_cache_hit(self) -> None:
        df = _make_config_df()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache_path = cache_dir / "Fashion-MNIST" / "config.parquet.gzip"
            cache_path.parent.mkdir(parents=True)
            df.to_parquet(cache_path, engine="pyarrow", compression="gzip")
            with patch.object(lcbench.urllib.request, "urlopen") as mock_urlopen:
                loaded = _read_lcbench_parquet(
                    dataset_name="Fashion-MNIST", stem="config", cache_dir=cache_dir
                )
            mock_urlopen.assert_not_called()
        pd.testing.assert_frame_equal(loaded, df)

    def test_read_parquet_cache_miss_downloads_and_caches(self) -> None:
        df = _make_config_df()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            raw = df.to_parquet(engine="pyarrow", compression="gzip")
            response = MagicMock()
            response.__enter__.return_value.read.return_value = raw
            with patch.object(
                lcbench.urllib.request, "urlopen", return_value=response
            ) as mock_urlopen:
                loaded = _read_lcbench_parquet(
                    dataset_name="Fashion-MNIST", stem="config", cache_dir=cache_dir
                )
            url = mock_urlopen.call_args.args[0]
            self.assertIn("Fashion-MNIST/config.parquet.gzip", url)
            self.assertEqual(
                mock_urlopen.call_args.kwargs["timeout"],
                lcbench.DEFAULT_LCBENCH_DOWNLOAD_TIMEOUT,
            )
            cache_path = cache_dir / "Fashion-MNIST" / "config.parquet.gzip"
            self.assertTrue(cache_path.exists())
            self.assertEqual(cache_path.read_bytes(), raw)
        pd.testing.assert_frame_equal(loaded, df)

    def test_read_parquet_corrupt_download_is_not_cached(self) -> None:
        # A truncated download or an HTML error body must not poison the cache:
        # the parse has to fail *before* anything is written.
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"<html>404</html>"
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch.object(lcbench.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(pyarrow.ArrowInvalid):
                    _read_lcbench_parquet(
                        dataset_name="Fashion-MNIST",
                        stem="config",
                        cache_dir=cache_dir,
                    )
            cache_path = cache_dir / "Fashion-MNIST" / "config.parquet.gzip"
            self.assertFalse(cache_path.exists())
            # No stray temp file is left behind in the cache directory either.
            self.assertEqual(list(cache_dir.rglob("*.tmp")), [])

    def test_load_lcbench_data(self) -> None:
        config_df, metrics_df = _make_config_df(), _make_metrics_df()
        for dtype in (torch.float, torch.double):
            with patch.object(
                lcbench,
                "_read_lcbench_parquet",
                side_effect=[config_df.copy(), metrics_df],
            ):
                data = load_lcbench_data(
                    "Fashion-MNIST", dtype=dtype, cache_dir=Path("/tmp/unused")
                )

            n_params = len(LCBENCH_PARAMETER_NAMES)
            self.assertEqual(data.parameters.shape, torch.Size([N_TRIALS, n_params]))
            self.assertEqual(data.metrics.shape, torch.Size([N_TRIALS, 50]))
            self.assertEqual(data.parameters.dtype, dtype)
            self.assertEqual(data.metrics.dtype, dtype)
            self.assertEqual(data.epochs.dtype, dtype)
            self.assertTrue(torch.equal(data.epochs, torch.arange(1, 51, dtype=dtype)))

            # Columns are subset and reordered, log-scale ones are logged.
            expected = config_df[list(LCBENCH_PARAMETER_NAMES)].copy()
            log_names = list(LCBENCH_LOG_SCALE_PARAMETER_NAMES)
            expected[log_names] = np.log(expected[log_names])
            self.assertAllClose(
                data.parameters, torch.from_numpy(expected.values).to(dtype)
            )

            # Epochs 0 and 51 are dropped from the 52 recorded epochs.
            expected_metrics = (
                metrics_df[DEFAULT_LCBENCH_METRIC_NAME]
                .unstack(level="epoch")
                .iloc[:, 1:-1]
            )
            self.assertAllClose(
                data.metrics, torch.from_numpy(expected_metrics.values).to(dtype)
            )

    def test_default_cache_dir(self) -> None:
        with patch.object(
            lcbench,
            "_read_lcbench_parquet",
            side_effect=[_make_config_df(), _make_metrics_df()],
        ) as mock_read:
            load_lcbench_data("Fashion-MNIST")
        self.assertEqual(
            mock_read.call_args.kwargs["cache_dir"], lcbench._default_cache_dir()
        )
        # Resolved lazily, so `$HOME` is read at call time rather than at import.
        self.assertEqual(
            lcbench._default_cache_dir(),
            Path(lcbench.DEFAULT_LCBENCH_CACHE_DIR).expanduser(),
        )

    def test_dataset_names(self) -> None:
        self.assertEqual(len(LCBENCH_DATASET_NAMES), 35)
        self.assertIn("Fashion-MNIST", LCBENCH_DATASET_NAMES)
        self.assertEqual(
            set(LCBENCH_LOG_SCALE_PARAMETER_NAMES) - set(LCBENCH_PARAMETER_NAMES),
            set(),
        )
