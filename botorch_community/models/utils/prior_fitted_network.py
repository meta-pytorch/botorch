#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import hashlib
import io
import os
import sys
from dataclasses import dataclass
from enum import Enum

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


DEFAULT_CACHE_DIR = "/tmp/botorch_pfn_models"
ACCEPT_LICENSE_ENV_VAR = "BOTORCH_PFN_ACCEPT_LICENSE"


@dataclass(frozen=True)
class ModelLicense:
    """License metadata for a downloadable pretrained model.

    Args:
        name: Human-readable license name, e.g. ``"Apache-2.0"``.
        url: Canonical URL where the license can be read.
        text_url: Optional URL of the raw license text; if given, a copy is
            saved next to the downloaded weights.
        requires_acceptance: If ``True``, the user must explicitly accept the
            license before the model is downloaded (see
            ``ensure_license_accepted``). If ``False``, only a notice is
            logged.
        attribution: Optional attribution string the license requires to be
            displayed when the model is used or redistributed.
    """

    name: str
    url: str
    text_url: str | None = None
    requires_acceptance: bool = False
    attribution: str | None = None


MODEL_LICENSES: dict[str, ModelLicense] = {
    ModelPaths.pfns4bo_hebo.value: ModelLicense(
        name="Apache-2.0",
        url="https://github.com/automl/PFNs4BO/blob/main/LICENSE",
        text_url="https://raw.githubusercontent.com/automl/PFNs4BO/main/LICENSE",
    ),
    ModelPaths.pfns4bo_bnn.value: ModelLicense(
        name="Apache-2.0",
        url="https://github.com/automl/PFNs4BO/blob/main/LICENSE",
        text_url="https://raw.githubusercontent.com/automl/PFNs4BO/main/LICENSE",
    ),
    "Prior-Labs/TabPFN-v2-reg": ModelLicense(
        name="Prior Labs License (Apache 2.0 with additional attribution)",
        url="https://huggingface.co/Prior-Labs/TabPFN-v2-reg",
        text_url="https://raw.githubusercontent.com/PriorLabs/TabPFN/main/LICENSE",
        requires_acceptance=True,
        attribution="Built with PriorLabs-TabPFN",
    ),
}


def _acceptance_marker_path(license: ModelLicense, cache_dir: str) -> str:
    key = hashlib.sha256(license.url.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f".license_accepted_{key}")


def save_license_copy(
    license: ModelLicense,
    cache_dir: str | None = None,
    proxies: dict[str, str] | None = None,
) -> str | None:
    """Save a copy of the license text into the model cache directory.

    Failures (e.g. no network) are logged and swallowed — the license URL is
    always available in ``license.url``.

    Args:
        license: The license whose text to save.
        cache_dir: The cache dir to use, defaulting to
            ``/tmp/botorch_pfn_models``.
        proxies: An optional dictionary mapping from network protocols to
            proxy addresses.

    Returns:
        The path of the saved license file, or ``None`` if unavailable.
    """
    if license.text_url is None:
        return None
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(license.url.encode()).hexdigest()[:16]
    license_path = os.path.join(cache_dir, f"LICENSE_{key}.txt")
    if os.path.exists(license_path):
        return license_path
    try:
        response = requests.get(license.text_url, proxies=proxies or None)
        response.raise_for_status()
        with open(license_path, "w") as f:
            f.write(response.text)
        return license_path
    except Exception as e:  # pragma: no cover - network-dependent
        logger.warning(f"Could not save a copy of the model license: {e}")
        return None


def ensure_license_accepted(
    license: ModelLicense,
    accept_license: bool = False,
    cache_dir: str | None = None,
) -> None:
    """Ensure the user has seen (and, if required, accepted) a model license.

    For licenses with ``requires_acceptance=False`` this logs a notice and
    returns. For licenses that require acceptance, acceptance can be given
    (in order of precedence) by:

    1. a previously recorded acceptance marker in ``cache_dir``;
    2. passing ``accept_license=True``;
    3. setting the environment variable ``BOTORCH_PFN_ACCEPT_LICENSE=1``;
    4. answering an interactive prompt (only if run in a terminal).

    Otherwise a ``RuntimeError`` is raised with instructions. Acceptance is
    recorded in ``cache_dir`` so it is requested at most once per machine.

    Args:
        license: The license to display and check.
        accept_license: If ``True``, the caller confirms the user accepts the
            license terms.
        cache_dir: Directory for the acceptance marker, defaulting to
            ``/tmp/botorch_pfn_models``.
    """
    notice = (
        f"This model is distributed under the {license.name} license: {license.url}"
    )
    if license.attribution is not None:
        notice += f' (attribution requirement: "{license.attribution}")'
    if not license.requires_acceptance:
        logger.info(notice)
        return

    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    marker = _acceptance_marker_path(license, cache_dir)
    if os.path.exists(marker):
        return

    accepted = accept_license or os.environ.get(ACCEPT_LICENSE_ENV_VAR, "").lower() in (
        "1",
        "true",
        "yes",
    )
    if not accepted and sys.stdin is not None and sys.stdin.isatty():
        answer = input(
            f"{notice}\nDo you accept the license terms? [y/N] "
        )  # pragma: no cover - interactive
        accepted = answer.strip().lower() in ("y", "yes")  # pragma: no cover
    if not accepted:
        raise RuntimeError(
            f"{notice}\nDownloading this model requires accepting its "
            f"license. Pass `accept_license=True`, or set the environment "
            f"variable `{ACCEPT_LICENSE_ENV_VAR}=1`."
        )
    with open(marker, "w") as f:
        f.write(license.url + "\n")
    logger.info(notice)
    save_license_copy(license, cache_dir=cache_dir)


def download_model(
    model_path: str | ModelPaths,
    proxies: dict[str, str] | None = None,
    cache_dir: str | None = None,
    accept_license: bool = False,
) -> nn.Module:
    """Download and load PFN model weights from a URL.

    Args:
        model_path: A string representing the URL of the model to load or a ModelPaths
            enum.
        proxies: An optional dictionary mapping from network protocols, e.g. ``http``,
            to proxy addresses.
        cache_dir: The cache dir to use, if not specified we will use
            ``/tmp/botorch_pfn_models``
        accept_license: If the model's license requires explicit acceptance
            (see ``MODEL_LICENSES``), pass ``True`` to confirm the user
            accepts the license terms.

    Returns:
        A PFN model.
    """
    if isinstance(model_path, ModelPaths):
        model_path = model_path.value

    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, model_path.split("/")[-1])

    license = MODEL_LICENSES.get(model_path)
    if license is not None:
        ensure_license_accepted(
            license, accept_license=accept_license, cache_dir=cache_dir
        )

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
        if license is not None:
            save_license_copy(license, cache_dir=cache_dir, proxies=proxies)
    else:
        # Load the model from cache
        model = torch.load(
            cache_path, weights_only=False, map_location=torch.device("cpu")
        )
        logger.debug("Model file loaded from cache: ", cache_path)

    return model
