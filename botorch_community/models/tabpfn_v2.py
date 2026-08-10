#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
BoTorch model wrapper for the TabPFN v2 tabular foundation model.

Built with PriorLabs-TabPFN.

``TabPFNv2Model`` adapts the pretrained TabPFN v2 regressor
[hollmann2025tabpfn]_ to the BoTorch ``Model`` interface by subclassing the
community ``PFNModel``. TabPFN v2 differs architecturally from the
``pfns4bo``-style PFNs that ``PFNModel`` targets (two-dimensional
feature/sample attention, a 5000-bucket full-support bar distribution, and a
``forward(x, y)`` interface that infers the train/test split from the length
of ``y``), so this subclass overrides the prediction path:

- the raw TabPFN v2 transformer is called sequence-first with the training
  block followed by the test block, standardized targets, and no TabPFN
  sklearn-style preprocessing/ensembling — inputs are expected in ``[0, 1]^d``
  as in Bayesian optimization, which keeps the posterior differentiable with
  respect to the test inputs (as required by, e.g., GIT-BO [yu2025gitbo]_);
- target standardization is handled internally: ``train_Y`` is standardized
  before the forward pass and the bar-distribution borders are mapped back to
  the raw scale, so the returned ``BoundedRiemannPosterior`` lives in the
  original units of ``train_Y``.

The pretrained weights are downloaded from Hugging Face
(``Prior-Labs/TabPFN-v2-reg``) on first use. They are distributed under the
Prior Labs License (Apache 2.0 with an additional attribution provision);
downloading requires accepting that license — see ``accept_license`` in
``download_tabpfn_v2_regressor``. Requires the ``tabpfn`` and
``huggingface_hub`` packages (``pip install tabpfn``).

References

.. [hollmann2025tabpfn]
    N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer,
    S. B. Hoo, R. T. Schirrmeister, F. Hutter. Accurate predictions on
    small data with a tabular foundation model. Nature, 2025.
.. [yu2025gitbo]
    R. T.-Y. Yu, C. Picard, F. Ahmed. GIT-BO: High-Dimensional Bayesian
    Optimization with Tabular Foundation Models. International Conference
    on Learning Representations, 2026. arXiv:2505.20685.

Contributor: rosenyu304
"""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.transforms.input import InputTransform
from botorch_community.models.prior_fitted_network import PFNModel
from botorch_community.models.utils.prior_fitted_network import (
    ensure_license_accepted,
    MODEL_LICENSES,
)
from torch import Tensor
from torch.nn import Module

TABPFN_V2_REG_REPO = "Prior-Labs/TabPFN-v2-reg"
TABPFN_V2_REG_FILE = "tabpfn-v2-regressor.ckpt"


def download_tabpfn_v2_regressor(
    accept_license: bool = False,
) -> tuple[Module, Module]:
    """Download and load the pretrained TabPFN v2 regressor.

    The checkpoint is fetched from Hugging Face
    (``Prior-Labs/TabPFN-v2-reg``) with local caching via
    ``huggingface_hub``. The weights are distributed under the Prior Labs
    License (Apache 2.0 with an additional attribution provision), which
    must be accepted before the download proceeds.

    Args:
        accept_license: Pass ``True`` to confirm the user accepts the model
            license terms. Alternatively set the environment variable
            ``BOTORCH_PFN_ACCEPT_LICENSE=1``. Without acceptance, a
            ``RuntimeError`` with instructions is raised.

    Returns:
        A two-element tuple containing:

        - The raw TabPFN v2 transformer module (in eval mode).
        - The full-support bar distribution with the bucket ``borders``.
    """
    ensure_license_accepted(
        MODEL_LICENSES[TABPFN_V2_REG_REPO], accept_license=accept_license
    )
    try:
        from huggingface_hub import hf_hub_download
        from tabpfn.model_loading import load_model
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TabPFNv2Model requires the `tabpfn` and `huggingface_hub` "
            "packages. Install them with `pip install tabpfn`."
        ) from e
    from pathlib import Path

    path = hf_hub_download(repo_id=TABPFN_V2_REG_REPO, filename=TABPFN_V2_REG_FILE)
    model, bar_distribution, _, _ = load_model(path=Path(path))
    model.eval()
    return model, bar_distribution


class TabPFNv2Model(PFNModel):
    """BoTorch model backed by the TabPFN v2 regressor.

    Built with PriorLabs-TabPFN.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        model: Module | None = None,
        bar_distribution: Module | None = None,
        train_Yvar: Tensor | None = None,
        constant_model_kwargs: dict[str, Any] | None = None,
        input_transform: InputTransform | None = None,
        accept_license: bool = False,
    ) -> None:
        """Initialize a TabPFNv2Model.

        Args:
            train_X: A ``n x d`` tensor of training features (for BO, in the
                unit cube).
            train_Y: A ``n x 1`` tensor of training observations, in raw
                units. Standardization is handled internally and the
                posterior is returned in raw units.
            model: An optional TabPFN v2 transformer. If ``None``, the
                pretrained regressor is downloaded (see
                ``download_tabpfn_v2_regressor``).
            bar_distribution: The bar distribution matching ``model``.
                Required if ``model`` is provided; ignored otherwise.
            train_Yvar: Observed variance of train_Y. Currently ignored.
            constant_model_kwargs: A dictionary of kwargs passed to the
                transformer in each forward pass (e.g.
                ``categorical_inds``).
            input_transform: A BoTorch input transform.
            accept_license: Pass ``True`` to accept the Prior Labs model
                license when the pretrained weights need to be downloaded.
        """
        if model is None:
            model, bar_distribution = download_tabpfn_v2_regressor(
                accept_license=accept_license
            )
        elif bar_distribution is None:
            raise ValueError(
                "bar_distribution must be provided when model is provided."
            )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            model=model,
            train_Yvar=train_Yvar,
            batch_first=False,
            constant_model_kwargs=constant_model_kwargs,
            input_transform=input_transform,
        )
        self.bar_distribution = bar_distribution
        # Standardization constants for the (fixed) training targets; the
        # bar-distribution borders are mapped back with these so the
        # posterior lives in the raw units of train_Y.
        self._y_mean = train_Y.mean()
        self._y_std = train_Y.std().clamp_min(1e-9)

    def pfn_predict(
        self,
        X: Tensor,
        train_X: Tensor,
        train_Y: Tensor,
        **forward_kwargs,
    ) -> Tensor:
        """Predict bucket probabilities with the TabPFN v2 transformer.

        Args:
            X: Test points of shape ``(b, q, d)``.
            train_X: Training features of shape ``(b, n, d)``.
            train_Y: Training targets of shape ``(b, n, 1)``, raw units.
            **forward_kwargs: Additional kwargs for the transformer.

        Returns:
            Probabilities of shape ``(b, q, num_buckets)`` over the
            (standardized-space) bar-distribution buckets.
        """
        train_Y_std = (train_Y - self._y_mean) / self._y_std
        # TabPFN v2 is sequence-first and infers the number of training
        # points from the length of y.
        x_full = torch.cat([train_X, X], dim=-2).transpose(0, 1)  # (n+q, b, d)
        y_train = train_Y_std.transpose(0, 1)  # (n, b, 1)
        logits = self.pfn(
            x=x_full.float(),
            y=y_train.float(),
            only_return_standard_out=True,
            **forward_kwargs,
        )  # (q, b, num_buckets)
        logits = logits.transpose(0, 1).to(X.dtype)  # (b, q, num_buckets)
        return logits.softmax(dim=-1)

    @property
    def borders(self) -> Tensor:
        """Bar-distribution borders, mapped back to the raw units of train_Y."""
        borders = self.bar_distribution.borders.to(
            dtype=self.train_X.dtype, device=self.train_X.device
        )
        return borders * self._y_std + self._y_mean
