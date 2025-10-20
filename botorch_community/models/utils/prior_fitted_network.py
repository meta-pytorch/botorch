#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import io
import os
from enum import Enum
from typing import Optional

from botorch.logging import logger

try:
    import requests
except ImportError:  # pragma: no cover
    raise ImportError(
        "The `requests` library is required to run `download_model`. "
        "You can install it using pip: `pip install requests`"
    )

try:
    import pfns4bo  # noqa: F401
except ImportError:  # pragma: no cover
    logger.warning(
        "pfns4bo is not installed, unable to automatically download PFN model."
    )

import torch
import torch.nn as nn


class ModelPaths(Enum):
    """Enum for PFN models"""

    pfns4bo_hebo = (
        "https://github.com/automl/PFNs4BO/raw/refs/heads/main/pfns4bo"
        "/final_models/model_hebo_morebudget_9_unused_features_3.pt.gz"
    )
    pfns4bo_bnn = (
        "https://github.com/automl/PFNs4BO/raw/refs/heads/main/pfns4bo"
        "/final_models/model_sampled_warp_simple_mlp_for_hpob_46.pt.gz"
    )


def download_model(
    model_path: str | ModelPaths,
    proxies: Optional[dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """Download and load PFN model weights from a URL.

    Args:
        model_path: A string representing the URL of the model to load or a ModelPaths
            enum.
        proxies: An optional dictionary mapping from network protocols, e.g. ``http``,
            to proxy addresses.
        cache_dir: The cache dir to use, if not specified we will use
            ``/tmp/botorch_pfn_models``

    Returns:
        A PFN model.
    """
    if isinstance(model_path, ModelPaths):
        model_path = model_path.value

    cache_dir = cache_dir if cache_dir is not None else "/tmp/botorch_pfn_models"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, model_path.split("/")[-1])

    if not os.path.exists(cache_path):
        # Download the model weights
        response = requests.get(model_path, proxies=proxies or None)
        response.raise_for_status()

        # Decompress the gzipped model weights
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            model = torch.load(gz, weights_only=False, map_location=torch.device("cpu"))

        # Save the model to cache
        torch.save(model, cache_path)
        logger.debug("Model file saved at: ", cache_path)
    else:
        # Load the model from cache
        model = torch.load(
            cache_path, weights_only=False, map_location=torch.device("cpu")
        )
        logger.debug("Model file loaded from cache: ", cache_path)

    return model
